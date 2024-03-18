# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
import time
import torch
from mylogger import logger
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from info.libs.models import load_model
from .base_model_worker import BaseModelWorker


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


class VLLMWorker(BaseModelWorker):
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_type: str,
            model_path: str,
            model_name: str,
            limit_worker_concurrency: int,
            gpu_memory_utilization: float,
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

        logger.info(
            f"Loading the model {self.model_name} on worker {worker_id}, worker type: vLLM worker..."
        )
        self.model = load_model(
            model_type=model_type,
            model_path=model_path,
            model_name=model_name,
            logger=logger,
            just_tokenizer=True,
            **kwargs,
        )

        engine_args = AsyncEngineArgs(model=model_path,
                                      tensor_parallel_size=torch.cuda.device_count(),
                                      trust_remote_code=True,
                                      gpu_memory_utilization=gpu_memory_utilization)
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.init_heart_beat()

    async def generate_stream_gate(self, request, request_id, prompt, generation_configs={}, **kwargs):
        self.call_ct += 1

        temperature = float(generation_configs.get("temperature", 1.0)) or self.model.generation_config.temperature
        top_p = float(generation_configs.get("top_p", 1.0)) or self.model.generation_config.top_p
        max_new_tokens = generation_configs.get("max_new_tokens", self.model.max_new_tokens)
        stop_str = generation_configs.get("stop", None)
        stop_token_ids = generation_configs.get("stop_token_ids", None) or []

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        for tid in stop_token_ids:
            if tid is not None:
                s = self.model.tokenizer.decode(tid)
                if s != "":
                    stop.add(s)

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            stop=list(stop),
            stop_token_ids=stop_token_ids,
            max_tokens=max_new_tokens
        )
        prompt, prompt_token_ids = self.model.build_chat_inputs(prompt=prompt)

        start = time.time()
        results_generator = self.llm_engine.generate(prompt, sampling_params, request_id)

        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            text_outputs = " ".join(text_outputs)

            partial_stop = any(is_partial_stop(text_outputs, i) for i in stop)
            # prevent yielding partial stop sequence
            if partial_stop:
                continue

            aborted = False
            if request and await request.is_disconnected():
                await self.llm_engine.abort(request_id)
                request_output.finished = True
                aborted = True
                for output in request_output.outputs:
                    output.finish_reason = "abort"

            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = sum(
                len(output.token_ids) for output in request_output.outputs
            )
            time_cost = time.time() - start
            ret = {"model_name": self.model.model_name,
                   "answer": text_outputs,
                   "history": [],
                   "time_cost": {"generation": f"{time_cost:.3f}s"},
                   "usage": {"prompt_tokens": prompt_tokens,
                             "generation_tokens": completion_tokens,
                             "total_tokens": prompt_tokens + completion_tokens,
                             "average_speed": f"{completion_tokens / time_cost:.3f} token/s"}
                   }
            yield json.dumps(ret, ensure_ascii=False).encode() + b"\0"

            if aborted:
                break


    async def generate_gate(self, **kwargs):
        async for x in self.generate_stream_gate(**kwargs):
            pass
        return json.loads(x[:-1].decode())
