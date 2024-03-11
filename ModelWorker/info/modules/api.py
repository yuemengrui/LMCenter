# *_*coding:utf-8 *_*
# @Author : YueMengRui
import uuid
import asyncio
from fastapi import APIRouter, Request
from info import logger, limiter, worker
from configs import API_LIMIT
from .protocol import ChatRequest, TokenCountRequest, TokenCountResponse
from fastapi.responses import JSONResponse, StreamingResponse
from info.utils.workers import acquire_worker_semaphore, release_worker_semaphore, create_background_tasks, \
    create_background_tasks_vllm, ModelWorker

router = APIRouter()


@router.api_route('/ai/worker/status', methods=['GET'], summary="worker status")
@limiter.limit(API_LIMIT['base'])
async def get_worker_status(request: Request):
    return JSONResponse(worker.get_status())


@router.api_route('/ai/worker/token_count', methods=['POST'], summary="token count")
@limiter.limit(API_LIMIT['token_count'])
async def count_token(request: Request,
                      req: TokenCountRequest
                      ):
    logger.info(req.dict())

    code, prompt_token_len, max_length, model_name = worker.model.check_token_len(req.prompt)

    return JSONResponse(TokenCountResponse(model_name=model_name,
                                           prompt=req.prompt,
                                           prompt_tokens=prompt_token_len,
                                           max_tokens=max_length,
                                           status='ok' if code else 'token_overflow').dict())


@router.api_route('/ai/worker/generate', methods=['POST'], summary="Generate")
@limiter.limit(API_LIMIT['chat'])
async def llm_chat(request: Request,
                   req: ChatRequest
                   ):
    logger.info(req.dict())

    await acquire_worker_semaphore(worker)

    if isinstance(worker, ModelWorker):
        if req.stream:
            generator = worker.generate_stream_gate(**req.dict())
            background_tasks = create_background_tasks(worker)
            return StreamingResponse(generator, background=background_tasks)
        else:
            output = await asyncio.to_thread(worker.generate_gate, **req.dict())
            release_worker_semaphore(worker)
            return JSONResponse(output)
    else:
        request_id = str(uuid.uuid4().hex)
        if req.stream:
            generator = worker.generate_stream_gate(request, request_id, **req.dict())
            background_tasks = create_background_tasks_vllm(worker.llm_engine, request_id, worker)
            return StreamingResponse(generator, background=background_tasks)
        else:
            output = await worker.generate_gate(request=request, request_id=request_id, **req.dict())
            release_worker_semaphore(worker)
            await worker.llm_engine.abort(request_id)
            return JSONResponse(output)
