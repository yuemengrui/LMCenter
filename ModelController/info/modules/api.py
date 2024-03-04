# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
import time
import httpx
import aiohttp
from fastapi import APIRouter, Request
from info import logger, limiter, controller
from configs import API_LIMIT, WORKER_API_TIMEOUT
from .protocol import TokenCountRequest, ErrorResponse, ChatRequest, WorkerRegisterRequest, ReceiveHeartBeatRequest
from fastapi.responses import JSONResponse, StreamingResponse
from info.utils.response_code import RET, error_map

router = APIRouter()


async def fetch_remote(url, payload=None):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3 * 3600)) as session:
        async with session.post(url, json=payload) as response:
            chunks = []
            if response.status != 200:
                ret = {
                    "errcode": RET.SERVERERR,
                    "errmsg": error_map[RET.SERVERERR]
                }
                return ret

            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        output = b"".join(chunks)

    return json.loads(output)


@router.api_route('/ai/worker_register', methods=['POST'], summary="Worker Register")
@limiter.limit(API_LIMIT['base'])
async def register_worker(request: Request,
                          req: WorkerRegisterRequest
                          ):
    logger.info(req.dict())

    controller.register_worker(req.worker_name, req.check_heart_beat, req.worker_status, req.multimodal)

    return JSONResponse({'msg': 'success'})


@router.api_route('/ai/receive_heart_beat', methods=['POST'], summary="receive_heart_beat")
@limiter.limit(API_LIMIT['base'])
async def receive_heart_beat(request: Request,
                             req: ReceiveHeartBeatRequest
                             ):
    logger.info(req.dict())

    exist = controller.receive_heart_beat(**req.dict())
    return JSONResponse({"exist": exist})


@router.api_route(path='/ai/llm/list', methods=['GET'], summary="获取支持的llm列表")
@limiter.limit(API_LIMIT['model_list'])
async def support_llm_list(request: Request):
    return JSONResponse({'data': [{'model_name': x} for x in controller.list_models()]})


@router.api_route('/ai/llm/token_count', methods=['POST'], summary="token count")
@limiter.limit(API_LIMIT['token_count'])
async def count_token(request: Request,
                      req: TokenCountRequest
                      ):
    logger.info(req.dict())

    if req.model_name not in controller.list_models():
        return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=u'Unsupported model!').dict())

    worker_addr = controller.get_worker_address(req.model_name)
    if worker_addr == '':
        return JSONResponse(
            ErrorResponse(errcode=RET.SERVERERR, errmsg=f"No available worker for {req.model_name}").dict())

    resp = await fetch_remote(url=worker_addr + '/ai/worker/token_count', payload=req.dict())

    return JSONResponse(resp)


@router.api_route('/ai/llm/chat', methods=['POST'], summary="Chat")
@limiter.limit(API_LIMIT['chat'])
async def llm_chat_simple(request: Request,
                          req: ChatRequest
                          ):
    start = time.time()
    logger.info(req.dict())

    if req.model_name not in controller.list_models():
        return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=u'Unsupported model!').dict())

    # token check
    worker_addr = controller.get_worker_address(req.model_name)
    if worker_addr == '':
        return JSONResponse(
            ErrorResponse(errcode=RET.SERVERERR, errmsg=f"No available worker for {req.model_name}").dict())

    resp = await fetch_remote(url=worker_addr + '/ai/worker/token_count', payload=req.dict())
    if resp['status'] != 'ok':
        return JSONResponse(ErrorResponse(errcode=RET.TOKEN_OVERFLOW,
                                          errmsg=error_map[
                                                     RET.TOKEN_OVERFLOW] + f"当前prompt token:{resp['prompt_tokens']} 支持的最大token:{resp['max_tokens']}").dict(),
                            status_code=413)

    if req.stream:
        return StreamingResponse(generate_completion_stream(payload=req.dict(), worker_addr=worker_addr, start=start),
                                 media_type="text/event-stream")
    else:
        resp = await fetch_remote(url=worker_addr + '/ai/worker/generate', payload=req.dict())
        resp['time_cost'].update({'total': f"{time.time() - start:.3f}s"})

        return JSONResponse(resp)


async def generate_completion_stream(payload, worker_addr: str, start):
    async with httpx.AsyncClient() as client:
        delimiter = b"\0"
        async with client.stream(
                "POST",
                worker_addr + "/ai/worker/generate",
                json=payload,
                timeout=WORKER_API_TIMEOUT,
        ) as response:
            buffer = b""
            async for raw_chunk in response.aiter_raw():
                buffer += raw_chunk
                while (chunk_end := buffer.find(delimiter)) >= 0:
                    chunk, buffer = buffer[:chunk_end], buffer[chunk_end + 1:]
                    if not chunk:
                        continue
                    resp = json.loads(chunk.decode())
                    resp['time_cost'].update({'total': f"{time.time() - start:.3f}s"})
                    yield json.dumps(resp, ensure_ascii=False)
