# *_*coding:utf-8 *_*
# @Author : YueMengRui
import asyncio
from mylogger import logger
from .model_worker import ModelWorker
from fastapi import BackgroundTasks

__all__ = ['build_worker', 'release_worker_semaphore', 'acquire_worker_semaphore', 'create_background_tasks',
           'create_background_tasks_vllm', 'ModelWorker']


def build_worker(worker_type=None, **kwargs):
    if worker_type == 'vllm':
        try:
            from .vllm_worker import VLLMWorker
            return VLLMWorker(**kwargs)
        except Exception as e:
            logger.error({'EXCEPTION': e})

    return ModelWorker(**kwargs)


def release_worker_semaphore(worker):
    worker.semaphore.release()


def acquire_worker_semaphore(worker):
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(worker):
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore, worker)
    return background_tasks


def create_background_tasks_vllm(llm_engine, request_id, worker):
    async def abort_request() -> None:
        await llm_engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore, worker)
    background_tasks.add_task(abort_request)
    return background_tasks
