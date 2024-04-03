# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import gc
import time
import torch
from typing import List
from copy import deepcopy
from .base_model import BaseModel, torch_gc, str_to_torch_dtype
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


class BaiChuan(BaseModel):

    def __init__(self,
                 model_path: str,
                 model_name: str,
                 logger=None,
                 device='cuda',
                 dtype=None,
                 lora_path='',
                 just_tokenizer=False,
                 dead_line=10 * 60,
                 **kwargs):
        self.dead_line = dead_line if isinstance(dead_line, int) else 10 * 60
        self.model_name = model_name
        self.model = None
        self.is_lora = False
        self.tokenizer = None
        self.generation_config = None
        self.device = None
        self.logger = logger
        self._load_model(model_path, lora_path, device, dtype, just_tokenizer)
        try:
            self.max_length = self.model.config.model_max_length
        except:
            self.max_length = 8 * 1024
        self.max_new_tokens = self.generation_config.max_new_tokens

        if self.logger:
            self.logger.info(str({'config': self.generation_config}) + '\n')
            self.logger.info(str({'max_length': self.max_length, 'max_new_tokens': self.max_new_tokens}) + '\n')

        # warmup
        if self.model:
            for _ in self.generate_stream('你好'):
                pass

    def _load_model(self, model_path, lora_path, device, dtype, just_tokenizer):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )
        self.generation_config = GenerationConfig.from_pretrained(model_path)

        if not just_tokenizer:
            if device == 'mps':
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True
                ).half().to('mps')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=str_to_torch_dtype(dtype),
                    device_map="auto",
                    trust_remote_code=True
                )

            if os.path.exists(lora_path):
                if self.logger:
                    self.logger.info(f"load lora from {lora_path}")
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                    self.is_lora = True

            self.device = self.model.device

    def check_token_len(self, prompt: str):
        code = True
        messages = [{'role': 'user', 'content': prompt}]
        prompt_token_len = self.token_counter(messages)
        if prompt_token_len > self.max_length:
            code = False

        return code, prompt_token_len, self.max_length, self.model_name

    def token_counter(self, messages: List):
        total_input, round_input = [], []
        for message in messages:
            content_tokens = self.tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [self.generation_config.user_token_id] + content_tokens
            elif message['role'] == 'assistant':
                round_input = [
                                  self.generation_config.assistant_token_id
                              ] + content_tokens + [
                                  self.generation_config.eos_token_id
                              ]
            total_input = total_input + round_input

        return len(total_input)

    def select_history(self, prompt, history, max_prompt_length):
        base_prompt_token_num = self.token_counter([{'role': 'user', 'content': prompt}])
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
            messages = [{'role': 'user', 'content': prompt}]

        total_input, round_input = [], []
        for i, message in enumerate(messages[::-1]):
            content_tokens = self.tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [self.generation_config.user_token_id] + content_tokens + round_input
                total_input = round_input + total_input
                round_input = []
            elif message['role'] == 'assistant':
                round_input = [
                                  self.generation_config.assistant_token_id
                              ] + content_tokens + [
                                  self.generation_config.eos_token_id
                              ] + round_input
            else:
                self.logger.warning(f"message role not supported yet: {message['role']}\n")

        total_input.append(self.generation_config.assistant_token_id)
        input_prompt = self.tokenizer.decode(total_input)
        if self.logger:
            self.logger.info(str({'prompt_len': len(total_input), 'prompt': input_prompt}) + '\n')
        return input_prompt, total_input

    @torch.inference_mode()
    def generate_stream(self, prompt: str, history=[], generation_configs={}, use_lora=False, **kwargs):

        if not (('max_new_tokens' in generation_configs) and (
                isinstance(generation_configs['max_new_tokens'], int)) and (
                        128 < generation_configs['max_new_tokens'] < self.max_length)):
            generation_configs.update({'max_new_tokens': self.max_new_tokens})

        generation_config = deepcopy(self.generation_config)
        generation_config.update(**generation_configs)

        max_prompt_length = self.max_length - generation_configs['max_new_tokens']

        if self.logger:
            self.logger.info(
                str({'max_prompt_length': max_prompt_length,
                     'generation_configs': generation_configs,
                     'genetation_config': generation_config}) + '\n' + str(
                    kwargs) + '\n')

        history = self.select_history(prompt, history, max_prompt_length)

        messages = []
        for his in history:
            messages.append({'role': 'user', 'content': his[0]})
            messages.append({'role': 'assistant', 'content': his[1]})

        messages.append({'role': 'user', 'content': prompt})

        input_prompt, prompt_token_ids = self.build_chat_inputs(messages=messages)
        prompt_tokens = len(prompt_token_ids)
        if self.logger:
            self.logger.info(str({'prompt_tokens': prompt_tokens, 'prompt_str_len': len(input_prompt),
                                  'prompt': input_prompt}) + '\n')

        start = time.time()
        last_token_time = time.time()
        token_latency = []
        if (not use_lora) and self.is_lora:
            with self.model.disable_adapter():
                for resp in self.model.chat(tokenizer=self.tokenizer,
                                            messages=messages,
                                            stream=True,
                                            generation_config=generation_config,
                                            ):
                    first_token_latency = time.time() - start
                    token_latency.append(time.time() - last_token_time)
                    token_latency.sort()
                    avg_token_latency = sum(token_latency) / len(token_latency)
                    last_token_time = time.time()
                    generation_tokens = len(self.tokenizer.encode(resp))
                    time_cost = time.time() - start

                    # exit
                    if time_cost > self.dead_line:
                        break

                    average_speed = f"{generation_tokens / time_cost:.3f} token/s"
                    yield {
                        "model_name": self.model_name,
                        "answer": resp,
                        "history": history,
                        "time_cost": {
                            "generation": f"{time_cost:.3f}s",
                            "first_token_latency": f"{first_token_latency * 1000:.2f}ms",
                            "token_latency": {
                                "min": f"{token_latency[0] * 1000:.2f}ms",
                                "max": f"{token_latency[-1] * 1000:.2f}ms",
                                "avg": f"{avg_token_latency * 1000:.2f}ms",
                            }
                        },
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "generation_tokens": generation_tokens,
                            "total_tokens": prompt_tokens + generation_tokens,
                            "average_speed": average_speed
                        }
                    }
        else:
            for resp in self.model.chat(tokenizer=self.tokenizer,
                                        messages=messages,
                                        stream=True,
                                        generation_config=generation_config,
                                        ):
                first_token_latency = time.time() - start
                token_latency.append(time.time() - last_token_time)
                token_latency.sort()
                avg_token_latency = sum(token_latency) / len(token_latency)
                last_token_time = time.time()
                generation_tokens = len(self.tokenizer.encode(resp))
                time_cost = time.time() - start

                # exit
                if time_cost > self.dead_line:
                    break

                average_speed = f"{generation_tokens / time_cost:.3f} token/s"
                yield {
                    "model_name": self.model_name,
                    "answer": resp,
                    "history": history,
                    "time_cost": {
                        "generation": f"{time_cost:.3f}s",
                        "first_token_latency": f"{first_token_latency * 1000:.2f}ms",
                        "token_latency": {
                            "min": f"{token_latency[0] * 1000:.2f}ms",
                            "max": f"{token_latency[-1] * 1000:.2f}ms",
                            "avg": f"{avg_token_latency * 1000:.2f}ms",
                        }
                    },
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "generation_tokens": generation_tokens,
                        "total_tokens": prompt_tokens + generation_tokens,
                        "average_speed": average_speed
                    }
                }

        torch_gc(self.device)
        gc.collect()
        if self.logger:
            self.logger.info({
                "model_name": self.model_name,
                "answer": resp,
                "history": history,
                "time_cost": {
                    "generation": f"{time_cost:.3f}s",
                    "first_token_latency": f"{first_token_latency * 1000:.2f}ms",
                    "token_latency": {
                        "min": f"{token_latency[0] * 1000:.2f}ms",
                        "max": f"{token_latency[-1] * 1000:.2f}ms",
                        "avg": f"{avg_token_latency * 1000:.2f}ms",
                    }
                },
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": generation_tokens,
                    "total_tokens": prompt_tokens + generation_tokens,
                    "average_speed": average_speed
                }
            })
