import json
import sqlite3
from uuid import uuid4
from datetime import datetime
from app.services.ticket_service import get_sqlite_conn,run_sqlite_write

# 避免每次请求都重复 CREATE TABLE support_sessions
_support_session_db_initialized=False

#1.初始化技术支持会话表
def init_support_session_db() ->None:
    global _support_session_db_initialized
    if _support_session_db_initialized:
        return
    def _init() ->None:
        global _support_session_db_initialized
        if _support_session_db_initialized:
            return
        with get_sqlite_conn() as conn:
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS support_sessions(
                    session_id TEXT PRIMARY KEY,
                    active_ticket_id TEXT,
                    last_intent TEXT,
                    stage TEXT,
                    last_diagnosis TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL)
                '''
            )
        _support_session_db_initialized=True
    run_sqlite_write(_init)
#2.生成技术支持对话ID
def generate_session_id() ->str:
    time_part=datetime.now().strftime("%Y%m%d%H%M%S")
    random_part=uuid4().hex[:6].upper()
    return f'S{time_part}{random_part}'

#3.返回当前时间字符串
def now_time_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#4.把数据库行转成普通 dict
def parsed_session_row(row:sqlite3.Row) ->dict|None:
    if not row:
        return None
    data=dict(row)
    if data.get('last_diagnosis'):
        try:
            data['last_diagnosis']=json.loads(data['last_diagnosis'])
        except json.JSONDecodeError:
            data['last_diagnosis']=None
    else:
        data['last_diagnosis']=None

    return data

#5.根据会话ID查询会话内容
def get_support_session_details(session_id:str|None) ->dict|None:
    init_support_session_db()
    session_id=str(session_id or '').strip()
    if not session_id:
        return None
    with get_sqlite_conn() as conn:
        row=conn.execute(
            '''
            SELECT * FROM support_sessions WHERE session_id=?
            ''',
            (session_id,),
        ).fetchone()

    return parsed_session_row(row)

#6.根据会话ID获取或者创建会话内容
def get_or_create_support_session(session_id:str|None=None) ->dict:
    init_support_session_db()
    session_id=str(session_id or '').strip() or generate_session_id()
    session=get_support_session_details(session_id)
    #有对话直接查询然后返回就好
    if session:
        return session
    #没有的话创建新的并插入信息
    current_time=now_time_text()
    def _write() ->None:
        with get_sqlite_conn() as conn:
            conn.execute(
                '''
                INSERT INTO support_sessions(
                    session_id,
                    active_ticket_id,
                    last_intent,
                    stage,
                    last_diagnosis,
                    created_at,
                    updated_at
                )
                VALUES(?,?,?,?,?,?,?)
                ''',
                (
                    session_id,
                    None,
                    None,
                    '会话已创建',
                    None,
                    current_time,
                    current_time
                ),
            )
    run_sqlite_write(_write)
    return get_support_session_details(session_id)

#7.更新会话状态
def update_support_session(
    session_id: str,
    active_ticket_id: str | None = None,
    last_intent: str | None = None,
    stage: str | None = None,
    last_diagnosis: dict | None = None,
) -> dict:
    init_support_session_db()
    session=get_or_create_support_session(session_id)

    new_active_ticket_id=active_ticket_id if active_ticket_id is not None else session.get('active_ticket_id')
    new_last_intent=last_intent if last_intent is not None else session.get('last_intent')
    new_stage=stage if stage is not None else session.get('stage')
    if last_diagnosis is not None:
        new_last_diagnosis=json.dumps(last_diagnosis,ensure_ascii=False)
    elif session.get('last_diagnosis') is not None:
        new_last_diagnosis=json.dumps(session.get('last_diagnosis'),ensure_ascii=False)
    else:
        new_last_diagnosis=None
    def _write() ->None:
        with get_sqlite_conn() as conn:
            conn.execute(
                '''
                UPDATE support_sessions
                SET 
                    active_ticket_id=?,
                    last_intent=?,
                    stage=?,
                    last_diagnosis=?,
                    updated_at=?
                WHERE session_id=?
                ''',
                (
                    new_active_ticket_id,
                    new_last_intent,
                    new_stage,
                    new_last_diagnosis,
                    now_time_text(),
                    session_id
                ),
            )
    run_sqlite_write(_write)
    return get_support_session_details(session_id)
















