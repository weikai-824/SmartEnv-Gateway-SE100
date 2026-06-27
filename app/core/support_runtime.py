'''把技术支持主链路从FastAPI默认线性池里隔离出去，放到专用线程池里执行，并加超时边界'''
import asyncio
from app.config.settings import Settings
from concurrent.futures import ThreadPoolExecutor,Future
from typing import Any,Callable
from functools import partial

#1.技术支持主链路专用线程池
_support_executor=ThreadPoolExecutor(
    max_workers=Settings.support_worker_threads,
    thread_name_prefix='support_worker'
)

#2.自定义异常，后续异常转换方便观察
class SupportTimeoutError(Exception):
    pass

#3.把同步主链路放到异步线性池里执行
async def run_support_in_worker(
        func:Callable[...,Any],
        *args:Any,
        timeout_seconds:int|None=None,
        **kwargs:Any
) ->Any:
    timeout_seconds=timeout_seconds or Settings.support_request_timeout
    loop=asyncio.get_running_loop()
    task=partial(func,*args,**kwargs)

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_support_executor,task),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError as e:
        raise SupportTimeoutError("技术支持主链路处理超时，请稍后重试") from e

#4.UI负责轮询future并持续yield处理中状态
def submit_support_to_worker(
        func:Callable[...,Any],
        *args:Any,
        **kwargs:Any
) ->Future:
    task=partial(func,*args,**kwargs)
    return _support_executor.submit(task)

#5.统一读取技术支持请求超时时间，避免 FastAPI 和 Gradio 两边各写一套配置。
def get_support_timeout_seconds(timeout_seconds: int | None = None) -> int:
    return timeout_seconds or Settings.support_request_timeout











