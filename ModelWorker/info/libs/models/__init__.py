# *_*coding:utf-8 *_*
# @Author : YueMengRui
from .model_baichuan import BaiChuan
from .model_chatglm3 import ChatGLM3

model_types = ['Baichuan', 'ChatGLM3']


def load_model(model_type, model_path, **kwargs):
    if model_type == 'Baichuan':
        model = BaiChuan(model_path, **kwargs)
    elif model_type == 'ChatGLM3':
        model = ChatGLM3(model_path, **kwargs)
    else:
        raise 'not support model:{}'.format(model_type)

    return model
