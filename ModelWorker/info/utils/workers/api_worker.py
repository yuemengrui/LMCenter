# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
import time
import httpx
from typing import Optional
from mylogger import logger
import torch
from transformers import set_seed
from info.utils.response_code import RET, error_map
from .base_model_worker import BaseModelWorker


class APIWorker(BaseModelWorker):
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_name: str,
            limit_worker_concurrency: int,
            token="",
            api_url="",
            seed: Optional[int] = None,
            multimodal=False,
            **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_name,
            limit_worker_concurrency,
            multimodal
        )

        self.api_url = api_url
        self.headers = {"Authorization": "Bearer " + token}

        self.seed = seed

        self.init_heart_beat()

    async def generate_stream_gate(self, prompt, generation_configs, history=None, stream=True, use_lora=False, **kwargs):
        start = time.time()
        self.call_ct += 1

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": True,
            **generation_configs
        }

        try:
            if self.seed is not None:
                set_seed(self.seed)
            async for output in generate_completion_stream(self.api_url, payload=payload, headers=self.headers, start=start):
                yield json.dumps(output, ensure_ascii=False).encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "errcode": RET.SERVERERR,
                "errmsg": "CUDA out of memory"
            }
            yield json.dumps(ret, ensure_ascii=False).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "errcode": RET.SERVERERR,
                "errmsg": error_map[RET.SERVERERR]
            }
            yield json.dumps(ret, ensure_ascii=False).encode() + b"\0"

    async def generate_gate(self, **kwargs):
        async for x in self.generate_stream_gate(**kwargs):
            pass
        return json.loads(x[:-1].decode())


async def generate_completion_stream(url, payload, headers, start, timeout=120):
    async with httpx.AsyncClient() as client:
        async with client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
        ) as response:
            answer = ""
            async for raw_chunk in response.aiter_raw():
                try:
                    raw_chunk_text = raw_chunk.decode().replace('data: ', '')
                    if "[DONE]" not in raw_chunk_text:
                        chunk = json.loads(raw_chunk_text)
                        answer += chunk['choices'][0]['delta']['content']
                        yield {"model_name": "",
                               "answer": answer,
                               "history": [],
                               "time_cost": {},
                               "usage": {}
                               }
                except:
                    pass
