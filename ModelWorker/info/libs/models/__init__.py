# *_*coding:utf-8 *_*
# @Author : YueMengRui
from .model_baichuan import BaiChuan
from .model_chatglm3 import ChatGLM3
from .model_qwen import Qwen2
from .model_qwenvl import QwenVL

model_types = ['Baichuan', 'ChatGLM3', 'Qwen2', 'QwenVL']


def load_model(model_type, **kwargs):
    if model_type == 'Baichuan':
        model = BaiChuan(**kwargs)
    elif model_type == 'ChatGLM3':
        model = ChatGLM3(**kwargs)
    elif model_type == 'Qwen2':
        model = Qwen2(**kwargs)
    elif model_type == 'QwenVL':
        model = QwenVL(**kwargs)
    else:
        raise f'Unsupported model type:{model_type}, only {model_types} are supported'

    return model
