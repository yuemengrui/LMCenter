# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from abc import ABC, abstractmethod
from transformers.utils.import_utils import is_torch_bf16_available


class BaseModel(ABC):

    @abstractmethod
    def check_token_len(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build_chat_inputs(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate_stream(self, **kwargs):
        raise NotImplementedError


def str_to_torch_dtype(dtype: str):
    if dtype is None:
        return None
    elif dtype == "float32":
        return torch.float32
    elif dtype == "bfloat16" and is_torch_bf16_available():
        return torch.bfloat16
    else:
        return torch.float16


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
