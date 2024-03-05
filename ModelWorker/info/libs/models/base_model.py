# *_*coding:utf-8 *_*
# @Author : YueMengRui
from abc import ABC, abstractmethod


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
