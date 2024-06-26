# *_*coding:utf-8 *_*
# @Author : YueMengRui
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Union


class ErrorResponse(BaseModel):
    errcode: int
    errmsg: str


class ChatRequest(BaseModel):
    prompt: Union[str, List]
    history: List = Field(default=[], description="历史记录")
    generation_configs: Dict = {}
    stream: bool = Field(default=True, description="是否流式输出")
    use_lora: bool = Field(default=False, description="是否使用lora模型生成结果")


class TokenCountRequest(BaseModel):
    prompt: Union[str, List]


class TokenCountResponse(BaseModel):
    object: str = 'token_count'
    model_name: str
    prompt: Union[str, List]
    prompt_tokens: int
    max_tokens: int
    status: str
