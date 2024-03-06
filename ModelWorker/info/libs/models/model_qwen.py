# *_*coding:utf-8 *_*
# @Author : YueMengRui
import gc
import time
import torch
from typing import List
from threading import Thread
from .base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation.utils import GenerationConfig
from transformers.utils.import_utils import is_torch_bf16_available


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


class Qwen2(BaseModel):

    def __init__(self, model_path: str, model_name: str, logger=None, device='cuda', dtype=None, just_tokenizer=False,
                 **kwargs):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.device = None
        self.logger = logger
        self._load_model(model_path, device, dtype, just_tokenizer)
        try:
            self.max_length = self.model.config.max_position_embeddings
        except:
            self.max_length = 32 * 1024
        self.max_new_tokens = 2048

        if self.logger:
            self.logger.info(str({'config': self.generation_config}) + '\n')
            self.logger.info(str({'max_length': self.max_length, 'max_new_tokens': self.max_new_tokens}) + '\n')

        # warmup
        if self.model:
            for _ in self.generate_stream('你好'):
                pass

    def _load_model(self, model_path, device, dtype, just_tokenizer):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )
        self.generation_config = GenerationConfig.from_pretrained(model_path)

        if not just_tokenizer:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=str_to_torch_dtype(dtype),
                device_map="auto",
                trust_remote_code=True
            )

            self.device = self.model.device

    def check_token_len(self, prompt: str):
        code = True
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        prompt_token_len = self.token_counter(messages)
        if prompt_token_len > self.max_length:
            code = False

        return code, prompt_token_len, self.max_length, self.model_name

    def token_counter(self, messages: List):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def select_history(self, prompt, history, max_prompt_length):
        base_prompt_token_num = self.token_counter(
            [{"role": "system", "content": "You are a helpful assistant."}, {'role': 'user', 'content': prompt}])
        true_history = []
        if history and base_prompt_token_num < max_prompt_length:
            for (old_query, old_response) in history[::-1]:
                history_token_num = self.token_counter(
                    [{'role': 'user', 'content': old_query}, {'role': 'assistant', 'content': old_response}])

                if base_prompt_token_num + history_token_num > max_prompt_length:
                    break
                else:
                    true_history.insert(0, [old_query, old_response])
                    base_prompt_token_num += history_token_num

        return true_history

    def build_chat_inputs(self, prompt: str = None, messages: List = []):
        if len(messages) == 0:
            messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {'role': 'user', 'content': prompt}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        if self.logger:
            self.logger.info(str({'prompt_len': len(model_inputs.input_ids[0]), 'prompt': prompt}) + '\n')

        return prompt, model_inputs.input_ids

    @torch.inference_mode()
    def generate_stream(self, prompt: str, history=[], generation_configs={}, stream=True, **kwargs):

        if not (('max_new_tokens' in generation_configs) and (
                isinstance(generation_configs['max_new_tokens'], int)) and (
                        2 < generation_configs['max_new_tokens'] < self.max_length)):
            generation_configs.update({'max_new_tokens': self.max_new_tokens})

        max_prompt_length = self.max_length - generation_configs['max_new_tokens']

        if self.logger:
            self.logger.info(
                str({'max_prompt_length': max_prompt_length,
                     'generation_configs': generation_configs}) + '\n' + str(
                    kwargs) + '\n')

        history = self.select_history(prompt, history, max_prompt_length)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for his in history:
            messages.append({'role': 'user', 'content': his[0]})
            messages.append({'role': 'assistant', 'content': his[1]})

        messages.append({'role': 'user', 'content': prompt})

        input_prompt, prompt_token_ids = self.build_chat_inputs(messages=messages)
        prompt_tokens = len(prompt_token_ids[0])
        if self.logger:
            self.logger.info(str({'prompt_tokens': prompt_tokens, 'prompt_str_len': len(input_prompt),
                                  'prompt': input_prompt}) + '\n')

        start = time.time()

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=self.model.generate,
                        kwargs=dict(inputs=prompt_token_ids, streamer=streamer,
                                    max_new_tokens=generation_configs['max_new_tokens']))
        thread.start()

        answer = ''
        for resp in streamer:
            answer += resp
            generation_tokens = len(self.tokenizer.encode(answer))
            time_cost = time.time() - start
            average_speed = f"{generation_tokens / time_cost:.3f} token/s"
            yield {"model_name": self.model_name,
                   "answer": answer,
                   "history": history,
                   "time_cost": {"generation": f"{time_cost:.3f}s"},
                   "usage": {"prompt_tokens": len(prompt_token_ids[0]), "generation_tokens": generation_tokens,
                             "total_tokens": len(prompt_token_ids[0]) + generation_tokens,
                             "average_speed": average_speed}}

        torch_gc(self.device)
        gc.collect()
        if self.logger:
            self.logger.info({"model_name": self.model_name,
                              "answer": answer,
                              "history": history,
                              "time_cost": {"generation": f"{time_cost:.3f}s"},
                              "usage": {
                                  "prompt_tokens": prompt_tokens,
                                  "generation_tokens": generation_tokens,
                                  "total_tokens": prompt_tokens + generation_tokens,
                                  "average_speed": average_speed}
                              })
