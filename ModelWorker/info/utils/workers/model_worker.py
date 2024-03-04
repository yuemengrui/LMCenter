# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from typing import Optional
from mylogger import logger
import torch
from transformers import set_seed
from info.utils.response_code import RET, error_map
from info.libs.models import load_model
from .base_model_worker import BaseModelWorker


class ModelWorker(BaseModelWorker):
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_type: str,
            model_path: str,
            model_name: str,
            limit_worker_concurrency: int,
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

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.model = load_model(
            model_type=model_type,
            model_path=model_path,
            model_name=model_name,
            logger=logger,
            **kwargs,
        )

        self.seed = seed

        self.init_heart_beat()

    def generate_stream_gate(self, **kwargs):
        self.call_ct += 1

        try:
            if self.seed is not None:
                set_seed(self.seed)
            for output in self.model.generate_stream(**kwargs):
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
