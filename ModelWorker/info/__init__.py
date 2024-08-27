# *_*coding:utf-8 *_*
import time
import asyncio
from mylogger import logger
from fastapi import BackgroundTasks
from fastapi.requests import Request
from starlette.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from info.utils.workers import build_worker, ModelWorker
from configs import ModelWorkerConfig

worker = build_worker(**ModelWorkerConfig)


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id=None):
    if isinstance(worker, ModelWorker):
        background_tasks = BackgroundTasks()
        background_tasks.add_task(release_worker_semaphore)
        return background_tasks
    else:
        async def abort_request() -> None:
            await worker.llm_engine.abort(request_id)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(release_worker_semaphore)
        background_tasks.add_task(abort_request)
        return background_tasks


limiter = Limiter(key_func=lambda *args, **kwargs: '127.0.0.1')


def app_registry(app):
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.middleware("http")
    async def api_time_cost(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        cost = time.time() - start
        logger.info(f'end request "{request.method} {request.url.path}" - {cost:.3f}s')
        return response

    app.mount("/static", StaticFiles(directory=f"static"), name="static")

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
    async def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " - ReDoc",
            redoc_js_url="/static/redoc.standalone.js",
        )

    from info.modules import register_router

    register_router(app)
