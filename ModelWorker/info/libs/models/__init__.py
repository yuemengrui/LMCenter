# *_*coding:utf-8 *_*
# @Author : YueMengRui
from .model_baichuan import BaiChuan

model_types = ['Baichuan']


def load_model(model_type, model_path, **kwargs):
    if model_type == 'Baichuan':
        model = BaiChuan(model_path, **kwargs)
    else:
        raise 'not support model:{}'.format(model_type)

    return model
