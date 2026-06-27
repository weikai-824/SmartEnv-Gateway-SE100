'''这个函数主要负责限流和降级'''

from contextlib import contextmanager
from threading import BoundedSemaphore,Lock
from app.config.settings import Settings

#1.定义同一时刻最多请求和进入高并发的阈值
max_support_concurrent_requests=Settings.support_max_concurrent_requests
degrade_concurrent_requests=Settings.support_degrade_concurrent_requests

#2.定义全局变量
_support_semaphore=BoundedSemaphore(max_support_concurrent_requests)
_active_lock=Lock()
_active_requests=0

#3.自定义异常，表示技术系统支持繁忙
class SupportBusyError(Exception):
    pass

@contextmanager
def support_load_guard():
    acquired=_support_semaphore.acquire(blocking=False)
    if not acquired:
        raise SupportBusyError('系统当前请求过多，请稍后重试')

    global _active_requests
    with _active_lock:
        _active_requests+=1
        current_active_requests=_active_requests
    degraded=current_active_requests >= degrade_concurrent_requests

    try:
        yield degraded
    finally:
        with _active_lock:
            _active_requests-=1
        _support_semaphore.release()













