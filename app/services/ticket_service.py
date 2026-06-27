import json
import sqlite3
from pathlib import Path
from uuid import uuid4
from app.config.settings import Settings
from typing import Any,Callable,Iterator
from contextlib import contextmanager
from datetime import datetime
from threading import RLock
import time
#创建保存数据库的路径
db_path=Path(Settings.data_dir) / 'tickets.db'
#SQLite同时只能有一个写事务，把本进程内的写操作串行化
sqlite_write_lock=RLock()
#只有第一次来查建表后面直接跳过
_ticket_db_initialized=False

#定义合法工单状态
valid_statuses = {
    "待处理",
    "待用户补充",
    "待工程师处理",
    "已解决",
    "已关闭",
    "已升级人工",
}
#定义合法优先级
valid_priorities = {
    "P0",
    "P1",
    "P2",
    "P3",
}
#返回当前时间字符串
def now_time_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#生成工单ID
def generate_ticket_id() ->str:
    time_part=datetime.now().strftime('%Y%m%d')
    random_part=uuid4().hex[:6].upper()
    ticket_id=f'T{time_part}{random_part}'
    return ticket_id

#判断SQLite是否因写锁繁忙而失败
def is_sqlite_busy_error(error:Exception) ->bool:
    if not isinstance(error,sqlite3.OperationalError):
        return False
    message=str(error).lower()
    return "database is locked" in message or "database table is locked" in message

#SQLite写入保护入口，保护写入函数可以安全的加锁，重试，失败处理
def run_sqlite_write(write_func:Callable[[],Any]) ->Any:
    max_retries=max(0,Settings.sqlite_write_max_retries)
    last_error:Exception|None=None
    for attempt in range(max_retries+1):
        try:
            with sqlite_write_lock:
                return write_func()
        except sqlite3.OperationalError as e:
            if not is_sqlite_busy_error(e):
                raise
            last_error=e
            if attempt >= max_retries:
                break
            sleep_seconds=Settings.sqlite_write_retry_base_seconds * (2 ** attempt)
            time.sleep(sleep_seconds)
    raise RuntimeError("SQLite 当前写入繁忙，请稍后重试") from last_error

#创建SQLite数据库连接
@contextmanager
def get_sqlite_conn() ->Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(
        db_path,
        timeout=Settings.sqlite_busy_timeout_ms / 1000
    )
    # 遇到写锁时等待一段时间，不要立刻报 database is locked
    conn.execute(f"PRAGMA busy_timeout={Settings.sqlite_busy_timeout_ms}")
    #改变数据库查询返回结果的方式
    conn.row_factory=sqlite3.Row
    #降低一点同步磁盘的严格要求，换一点性能
    conn.execute('PRAGMA synchronous = NORMAL')
    #使用WAL模式，提升读写并放能力
    conn.execute('PRAGMA journal_mode = WAL')
    #开启外键约束
    conn.execute('PRAGMA foreign_keys=ON')

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

#初始化工单表和工单备注表
def init_ticket_db() ->None:
    global _ticket_db_initialized
    if _ticket_db_initialized:
        return
    def _init() ->None:
        global _ticket_db_initialized
        if _ticket_db_initialized:
            return
        with get_sqlite_conn() as conn:
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS tickets(
                ticket_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                problem_type TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT,
                missing_information TEXT,
                source_query TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
                '''
            )
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS ticket_notes(
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id))
            
                '''
            )
            _ticket_db_initialized=True
    run_sqlite_write(_init)


#定义一个通用校验字段
def check_choice(value:str,valid_values:set[str],field_name:str) ->str:
    value=str(value or '').strip()
    if value not in valid_values:
        raise ValueError(f'{field_name}非法:{value}')
    return value

#解析缺失信息
def load_missing_information(raw:str|None) ->list[str]:
    if not raw:
        return []
    try:
        data=json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data,list) else []

#把 SQLite 查询结果转成普通 dict
def row_to_dict(row:sqlite3.Row,notes:list[sqlite3.Row]) ->dict:
    ticket=dict(row)
    ticket['missing_information']=load_missing_information(ticket.get('missing_information'))
    ticket['notes']=[dict(note) for note in notes or []]
    return ticket

#查询工单详情，包括工单基础信息和工单备注表
def get_ticket_detail(ticket_id:str) ->dict:
    init_ticket_db()
    ticket_id=str(ticket_id or '').strip()
    if not ticket_id:
        raise ValueError('ticket_id 不能为空')
    with get_sqlite_conn() as conn:
        ticket_row=conn.execute(
            'SELECT * FROM tickets WHERE ticket_id=?',
            (ticket_id,),
        ).fetchone()
        if not ticket_row:
            raise FileNotFoundError(f'工单不存在:{ticket_id}')
        note_rows=conn.execute(
            '''
            SELECT note_id,note,created_at
            FROM ticket_notes
            WHERE ticket_id=?
            ORDER BY note_id ASC 
            ''',
            (ticket_id,),
        ).fetchall()
        return row_to_dict(ticket_row,note_rows)

#向工单表里插入信息
def create_ticket(
    title:str,
    problem_type:str|None=None,
    priority:str='P2',
    summary:str='',
    missing_information:list[str]|None=None,
    source_query:str|None=None,
    status:str='待处理'
) ->dict:
    init_ticket_db()
    title=str(title or '').strip()
    if not title:
        raise ValueError('title不能为空')
    #校验优先级
    priority=check_choice(value=priority,valid_values=valid_priorities,field_name='priority')
    status=check_choice(value=status,valid_values=valid_statuses,field_name='status')
    ticket_id=generate_ticket_id()
    current_time=now_time_text()
    def _write()->None:
        with get_sqlite_conn() as conn:
            conn.execute(
                '''
                INSERT INTO tickets(
                    ticket_id,
                    title,
                    problem_type,
                    priority,
                    status,
                    summary,
                    missing_information,
                    source_query,
                    created_at,
                    updated_at
                )
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                ''',
                (ticket_id,
                    title,
                    problem_type,
                    priority,
                    status,
                    summary,
                    json.dumps(missing_information or [], ensure_ascii=False),
                    source_query,
                    current_time,
                    current_time,),
            )
    run_sqlite_write(_write)
    return get_ticket_detail(ticket_id=ticket_id)

#更新工单状态
def update_ticket_status(ticket_id:str,status:str) ->dict:
    init_ticket_db()
    ticket_id=str(ticket_id or '').strip()
    if not ticket_id:
        raise ValueError('ticket_id不能为空')
    status=check_choice(value=status,valid_values=valid_statuses,field_name='status')
    def _write() ->None:
        with get_sqlite_conn() as conn:
            result=conn.execute(
                '''
                UPDATE tickets
                SET status=?,updated_at=?
                WHERE ticket_id=?
                ''',
                (status,now_time_text(),ticket_id)
            )
            if result.rowcount == 0:
                raise FileNotFoundError(f'工单不存在:{ticket_id}')
    run_sqlite_write(_write)
    return get_ticket_detail(ticket_id)

#向工单备注表里插入追加信息
def add_ticket_note(ticket_id: str, note: str) ->dict:
    init_ticket_db()
    ticket_id=str(ticket_id or '').strip()
    note=str(note or '').strip()
    if not ticket_id:
        raise ValueError('ticket_id不能为空')
    if not note:
        raise ValueError('note不能为空')
    #确保工单存在
    get_ticket_detail(ticket_id)
    def _write()->None:
        with get_sqlite_conn() as conn:
            conn.execute(
                '''
                INSERT INTO ticket_notes(ticket_id,note,created_at)
                VALUES (?,?,?)
                ''',
                (ticket_id,note,now_time_text()),
            )
            conn.execute(
                '''
                UPDATE tickets
                SET updated_at=?
                WHERE ticket_id=?
                ''',
                (now_time_text(),ticket_id),
            )
    run_sqlite_write(_write)
    return get_ticket_detail(ticket_id)

