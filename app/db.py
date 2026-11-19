"""PostgreSQL 访问层：schema 初始化、快照事务、溯源函数、数据维护。"""
from __future__ import annotations

import random
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, Callable, Iterator

import psycopg
from psycopg.rows import dict_row

from .config import settings
from .sql_rewrite import PROV_ADD, PROV_AGG
from .sql_validate import TABLE_SCHEMA

# 会话级溯源函数与聚合（创建在 pg_temp，事务结束随临时对象释放）
_PROV_SFUNC = f"""
CREATE OR REPLACE FUNCTION pg_temp.{PROV_ADD}(state jsonb, item jsonb)
RETURNS jsonb
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
  result jsonb := COALESCE(state, '{{}}'::jsonb);
  tk text;
  ik text;
  cur int;
  old_table jsonb;
  new_table jsonb;
BEGIN
  IF item IS NULL THEN
    RETURN result;
  END IF;
  FOR tk IN SELECT jsonb_object_keys(item) LOOP
    old_table := COALESCE(result -> tk, '{{}}'::jsonb);
    new_table := old_table;
    FOR ik IN SELECT jsonb_object_keys(item -> tk) LOOP
      cur := COALESCE((old_table ->> ik)::int, 0)
             + (item -> tk ->> ik)::int;
      -- 用 || 重建该表的计数对象（同名键覆盖为累加后的新值）
      new_table := new_table || jsonb_build_object(ik, cur);
    END LOOP;
    result := result || jsonb_build_object(tk, new_table);
  END LOOP;
  RETURN result;
END;
$$
"""

_PROV_FINAL = f"""
CREATE OR REPLACE FUNCTION pg_temp.{PROV_AGG}_final(state jsonb)
RETURNS jsonb
LANGUAGE sql IMMUTABLE AS $$
  SELECT COALESCE(
    jsonb_agg(
      jsonb_build_object('t', t, 'id', id, 'n', n)
      ORDER BY t, id
    ),
    '[]'::jsonb
  )
  FROM (
    SELECT tk.key AS t,
           (ik.key)::bigint AS id,
           (ik.value)::int AS n
    FROM jsonb_each(COALESCE(state, '{{}}'::jsonb)) tk,
         jsonb_each(tk.value) ik
  ) flat
$$
"""

_PROV_AGGREGATE_FINAL = f"""
CREATE AGGREGATE pg_temp.{PROV_AGG}(jsonb) (
  SFUNC = pg_temp.{PROV_ADD},
  STYPE = jsonb,
  FINALFUNC = pg_temp.{PROV_AGG}_final,
  INITCOND = '{{}}'
)
"""

# 不带 finalfunc 的聚合：输出仍是嵌套计数对象，供 CTE/派生表向外继续合并
_PROV_AGGREGATE_STATE = f"""
CREATE AGGREGATE pg_temp.{PROV_AGG}_state(jsonb) (
  SFUNC = pg_temp.{PROV_ADD},
  STYPE = jsonb,
  INITCOND = '{{}}'
)
"""

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS samples (
      id    INTEGER PRIMARY KEY,
      batch VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS results (
      id        INTEGER PRIMARY KEY,
      sample_id INTEGER NOT NULL REFERENCES samples(id),
      value     NUMERIC
    )
    """,
]


def admin_connect(autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(settings.admin_dsn, autocommit=autocommit)


def query_connect() -> psycopg.Connection:
    # 查询连接默认事务块，由 execute_lineage_query 统一提交/回滚
    return psycopg.connect(settings.query_dsn)


def initialize_schema() -> None:
    """创建物理表（幂等）。"""
    with admin_connect(autocommit=True) as conn:
        for stmt in DDL_STATEMENTS:
            conn.execute(stmt)


def count_rows() -> dict[str, int]:
    with admin_connect(autocommit=True) as conn:
        out = {}
        for table in TABLE_SCHEMA:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                out[table] = cur.fetchone()[0]
    return out


class RowLimitExceeded(ValueError):
    """新增行会超过该表的行数上限（更新既有行不受限）。"""


class WriteBusy(RuntimeError):
    """短时间内没能拿到该表的写锁；请求未做任何改动，可重试。"""


# 每张表一把事务级咨询锁：同一张表的写入串行化，不同表互不阻塞。
# 固定常量而非哈希，避免依赖 PostgreSQL 内部哈希函数的实现细节。
_TABLE_WRITE_LOCKS = {
    "samples": 0x5A51_0001,
    "results": 0x5A51_0002,
}

# 写锁用非阻塞 try-lock + 退避重试，而不是阻塞等待：
# 阻塞等待要依赖 PostgreSQL 后端之间的信号唤醒；连接池/受限环境里
# 一旦唤醒丢失，请求会无限悬挂。非阻塞方式把等待放在应用侧，可设上限。
_LOCK_WAIT_SECONDS = 10.0
_LOCK_RETRY_MIN_SECONDS = 0.005
_LOCK_RETRY_MAX_SECONDS = 0.1


def _try_acquire_table_write_lock(
    conn: psycopg.Connection, table: str
) -> bool:
    """尝试在当前事务内获取该表的写锁（不阻塞）。

    成功时事务提交/回滚即自动释放；失败时未持有任何锁。
    """
    return bool(
        conn.execute(
            "SELECT pg_try_advisory_xact_lock(%s)", (_TABLE_WRITE_LOCKS[table],)
        ).fetchone()[0]
    )


def _with_table_write_locks(
    tables: tuple[str, ...], work: Callable[[psycopg.Connection], Any]
) -> Any:
    """按给定顺序加表写锁后执行 ``work``，同一事务提交。

    锁的获取一律非阻塞；任一锁没拿到就回滚本事务、应用侧退避后整套重试，
    因此不会持有部分锁等待，也就不可能和别的写入互相等成死锁。
    重试有上限，超过则抛 WriteBusy（此时没有任何数据被改动）。

    名额判定与写入必须在同一条事务内完成：任何异常（含超限）都回滚，
    不会留下「已入库却返回 422」的半成品。不同 ID 并发争用同一张表的
    最后一个名额时由写锁串行化，只有一个能成功。
    """
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    delay = _LOCK_RETRY_MIN_SECONDS
    while True:
        with admin_connect() as conn:
            acquired_all = True
            for table in tables:
                if not _try_acquire_table_write_lock(conn, table):
                    acquired_all = False
                    break
            if acquired_all:
                try:
                    result = work(conn)
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
                return result
            # 没拿全锁：结束本事务（未持有事务级锁）后应用侧退避重试
            conn.rollback()
        if time.monotonic() >= deadline:
            raise WriteBusy(
                f"表 {'/'.join(tables)} 正在被其他写入占用，请稍后重试"
            )
        # 加抖动，避免多个等待者同时醒来再次撞车
        time.sleep(delay * (0.5 + random.random()))
        delay = min(delay * 2, _LOCK_RETRY_MAX_SECONDS)


def _locked_upsert(
    table: str, row_id: int, write: Callable[[psycopg.Connection], None]
) -> None:
    """在表级写锁保护下执行一次 upsert（先判定名额，再写入）。"""

    def work(conn: psycopg.Connection) -> None:
        _reserve_slot(conn, table, row_id)
        write(conn)

    _with_table_write_locks((table,), work)


def _reserve_slot(conn: psycopg.Connection, table: str, row_id: int) -> None:
    """在已持有表写锁的事务内做「更新 or 新增」判定。

    id 已存在时直接放行（按更新处理，即便表已超限也允许）；否则若新增
    会使行数超过上限，抛出 RowLimitExceeded，调用方随事务回滚，不会
    留下半成品数据。本函数必须在持有 ``_try_acquire_table_write_lock``
    的事务内调用，否则并发新增时计数与插入之间会出现竞态，导致超卖名额。
    """
    exists = conn.execute(
        f"SELECT 1 FROM {table} WHERE id = %s", (row_id,)
    ).fetchone()
    if exists is not None:
        return

    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if total >= settings.max_rows:
        raise RowLimitExceeded(
            f"表 {table} 已有 {total} 行，达到上限 {settings.max_rows} 行，"
            f"不能新增 id={row_id}；更新既有 ID 不受此限制"
        )


@contextmanager
def snapshot_transaction() -> Iterator[
    tuple[psycopg.Connection, str, dict[str, str]]
]:
    """开启一个固定快照事务，返回 (连接, snapshot_id, 物理表->临时表映射)。

    临时表 ON COMMIT DROP：事务提交或回滚后自动释放；同一事务内所有语句
    都基于这一份快照，期间其他事务的增改删不影响本次结果。
    """
    snapshot_id = uuid.uuid4().hex
    snapshot_map = {
        table: f"{settings.temp_prefix}_{snapshot_id}_{table}"
        for table in TABLE_SCHEMA
    }
    conn = query_connect()
    try:
        # REPEATABLE READ：事务内所有读都取事务开始时的同一快照
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        # 临时表与 pg_temp 对象优先解析（溯源函数调用本身已显式限定）
        conn.execute('SET search_path TO pg_temp, public')
        for table, temp in snapshot_map.items():
            conn.execute(
                f"CREATE TEMP TABLE {temp} (LIKE {table} INCLUDING ALL) "
                f"ON COMMIT DROP"
            )
            conn.execute(f"INSERT INTO {temp} SELECT * FROM {table}")
        conn.execute(_PROV_SFUNC)
        conn.execute(_PROV_FINAL)
        conn.execute(_PROV_AGGREGATE_FINAL)
        conn.execute(_PROV_AGGREGATE_STATE)
        yield conn, snapshot_id, snapshot_map
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_snapshot_contents(
    conn: psycopg.Connection, snapshot_map: dict[str, str]
) -> dict[str, dict[int, dict[str, Any]]]:
    """读取快照中每张表的全部行内容，用于溯源回填。"""
    contents: dict[str, dict[int, dict[str, Any]]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        for table, temp in snapshot_map.items():
            cur.execute(f"SELECT * FROM {temp}")
            contents[table] = {row["id"]: row for row in cur.fetchall()}
    return contents


def execute_sql(conn: psycopg.Connection, sql: str) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql)
        return cur.fetchall()


# --------------------------------------------------------------------------- #
# 数据维护接口（各自短事务，立即生效，不影响进行中的旧快照事务）
# --------------------------------------------------------------------------- #
def upsert_sample(sample_id: int, batch: str) -> None:
    """新增或修改样本。

    名额按 samples 表单独计算：仅新增且表已满 500 行时拒绝；
    更新既有 ID 始终允许（即使历史数据已超限）。整个判定与写入在
    同一事务、同一把表级写锁内完成，失败则整体回滚。
    """

    def write(conn: psycopg.Connection) -> None:
        conn.execute(
            "INSERT INTO samples(id, batch) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET batch = EXCLUDED.batch",
            (sample_id, batch),
        )

    _locked_upsert("samples", sample_id, write)


def delete_sample(sample_id: int) -> None:
    """删除样本及其检测行。

    同时持有 samples / results 两把表写锁（固定顺序），与并发的
    upsert/delete 串行化，不会出现「删到一半被计数」的中间态。
    """

    def work(conn: psycopg.Connection) -> None:
        conn.execute("DELETE FROM results WHERE sample_id = %s", (sample_id,))
        conn.execute("DELETE FROM samples WHERE id = %s", (sample_id,))

    _with_table_write_locks(("samples", "results"), work)


def upsert_result(
    result_id: int, sample_id: int, value: float | int | None
) -> None:
    """新增或修改检测行；名额按 results 表单独计算（规则同 upsert_sample）。"""

    def write(conn: psycopg.Connection) -> None:
        # 显式检查外键，给出可读错误
        exists = conn.execute(
            "SELECT 1 FROM samples WHERE id = %s", (sample_id,)
        ).fetchone()
        if not exists:
            raise ValueError(f"sample_id={sample_id} 不存在")
        try:
            conn.execute(
                "INSERT INTO results(id, sample_id, value) VALUES (%s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET sample_id = EXCLUDED.sample_id, "
                "value = EXCLUDED.value",
                (result_id, sample_id, value),
            )
        except psycopg.errors.ForeignKeyViolation as exc:
            # 检查与插入之间样本被并发删除：整条事务回滚，按 422 处理
            raise ValueError(f"sample_id={sample_id} 不存在") from exc

    _locked_upsert("results", result_id, write)


def delete_result(result_id: int) -> None:
    def work(conn: psycopg.Connection) -> None:
        conn.execute("DELETE FROM results WHERE id = %s", (result_id,))

    _with_table_write_locks(("results",), work)


def _jsonable(value: Any) -> Any:
    """把 PostgreSQL 类型转成可 JSON 序列化的普通值。"""
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return value


def list_table(table: str) -> list[dict[str, Any]]:
    if table not in TABLE_SCHEMA:
        raise ValueError(f"未知表 {table}")
    with admin_connect(autocommit=True) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT * FROM {table} ORDER BY id")
            return [
                {k: _jsonable(v) for k, v in row.items()}
                for row in cur.fetchall()
            ]


def reset_and_seed(rows: dict[str, list[dict[str, Any]]]) -> None:
    """整表重置（用于初始化/演示）。"""
    with admin_connect() as conn:
        conn.execute("TRUNCATE results, samples RESTART IDENTITY")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO samples(id, batch) VALUES (%(id)s, %(batch)s)",
                rows["samples"],
            )
            cur.executemany(
                "INSERT INTO results(id, sample_id, value) "
                "VALUES (%(id)s, %(sample_id)s, %(value)s)",
                rows["results"],
            )
        conn.commit()
