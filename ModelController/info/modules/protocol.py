# *_*coding:utf-8 *_*
# @Author : YueMengRui
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Union


class ErrorResponse(BaseModel):
    errcode: int
    errmsg: str


class ChatRequest(BaseModel):
    model_name: str = Field(default=None, description="模型名称")
    prompt: str
    history: List = Field(default=[], description="历史记录")
    generation_configs: Dict = {}
    stream: bool = Field(default=True, description="是否流式输出")
    use_lora: bool = Field(default=False, description="是否使用lora模型生成结果")


class WorkerRegisterRequest(BaseModel):
    worker_name: str
    check_heart_beat: bool
    worker_status: Dict
    multimodal: bool = Field(default=False, description="是否多模态")


class ReceiveHeartBeatRequest(BaseModel):
    worker_name: str
    queue_length: int


class TokenCountRequest(BaseModel):
    model_name: str = Field(default=None, description="模型名称")
    prompt: str
