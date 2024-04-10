# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import gc
import time
import torch
import base64
from io import BytesIO
from copy import deepcopy
from typing import List, Tuple
from peft import PeftModel
from PIL import Image
from .base_model import BaseModel, torch_gc, str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation.utils import GenerationConfig


def pil_base64(image: Image):
    img_buffer = BytesIO()
    image.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


class QwenVL(BaseModel):

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
            self.max_length = self.model.config.max_position_embeddings
        except:
            self.max_length = 8 * 1024

        self.max_new_tokens = self.model.generation_config.max_new_tokens
        self.ENDOFTEXT = "<|endoftext|>"
        self.IMSTART = "<|im_start|>"
        self.IMEND = "<|im_end|>"

        if self.logger:
            self.logger.info(str({'config': self.generation_config}) + '\n')
            self.logger.info(str({'max_length': self.max_length, 'max_new_tokens': self.max_new_tokens}) + '\n')

        # warmup
        # if self.model:
        #     for _ in self.generate_stream('你好'):
        #         pass

    def _load_model(self, model_path, lora_path, device, dtype, just_tokenizer):

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
            self.model.generation_config = self.generation_config

            if os.path.exists(lora_path):
                if self.logger:
                    self.logger.info(f"load lora from {lora_path}")
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                    self.is_lora = True

            self.device = self.model.device

    def check_token_len(self, prompt: List):
        code = True
        query = self.tokenizer.from_list_format(prompt)

        raw_text, context_tokens, history = self.make_context(query=query)

        prompt_token_len = len(context_tokens)

        if prompt_token_len > self.max_length:
            code = False

        return code, prompt_token_len, self.max_length, self.model_name

    def token_counter(self, prompt):
        return len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])

    def build_chat_inputs(self, prompt: List):
        query = self.tokenizer.from_list_format(prompt)

        raw_text, context_tokens, history = self.make_context(query=query)

        if self.logger:
            self.logger.info(str({'prompt_len': len(context_tokens), 'prompt': query}) + '\n')

        return prompt, context_tokens

    def make_context(self,
                     query: str,
                     history: List[Tuple[str, str]] = None,
                     system: str = "You are a helpful assistant.",
                     max_window_size: int = 6144,
                     chat_format: str = "chatml",
                     ):
        if history is None:
            history = []

        true_history = []

        if chat_format == "chatml":
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            im_start_tokens = [self.tokenizer.im_start_id]
            im_end_tokens = [self.tokenizer.im_end_id]
            nl_tokens = self.tokenizer.encode("\n")

            def _tokenize_str(role, content):
                return f"{role}\n{content}", self.tokenizer.encode(
                    role, allowed_special=set(self.tokenizer.IMAGE_ST)
                ) + nl_tokens + self.tokenizer.encode(content, allowed_special=set(self.tokenizer.IMAGE_ST))

            system_text, system_tokens_part = _tokenize_str("system", system)
            system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

            raw_text = ""
            context_tokens = []

            for turn_query, turn_response in reversed(history):
                query_text, query_tokens_part = _tokenize_str("user", turn_query)
                query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
                if turn_response is not None:
                    response_text, response_tokens_part = _tokenize_str(
                        "assistant", turn_response
                    )
                    response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                    next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                    prev_chat = (
                        f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                    )
                else:
                    next_context_tokens = nl_tokens + query_tokens + nl_tokens
                    prev_chat = f"\n{im_start}{query_text}{im_end}\n"

                current_context_size = (
                        len(system_tokens) + len(next_context_tokens) + len(context_tokens)
                )
                if current_context_size < max_window_size:
                    context_tokens = next_context_tokens + context_tokens
                    raw_text = prev_chat + raw_text
                    true_history.insert(0, [turn_query, turn_response])
                else:
                    break

            context_tokens = system_tokens + context_tokens
            raw_text = f"{im_start}{system_text}{im_end}" + raw_text
            context_tokens += (
                    nl_tokens
                    + im_start_tokens
                    + _tokenize_str("user", query)[1]
                    + im_end_tokens
                    + nl_tokens
                    + im_start_tokens
                    + self.tokenizer.encode("assistant")
                    + nl_tokens
            )
            raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        elif chat_format == "raw":
            raw_text = query
            context_tokens = self.tokenizer.encode(raw_text)
        else:
            raise NotImplementedError(f"Unknown chat format {chat_format!r}")

        return raw_text, context_tokens, true_history

    @torch.inference_mode()
    def generate_stream(self, prompt: List, history=[], generation_configs={}, use_lora=False, **kwargs):
        """
        :param prompt: [
            {'image': 'image local path or url'},
            {'text': ''},
        ]
        :param history:
        :param generation_configs:
        :param use_lora:
        :param kwargs:
        :return:
        """

        if not isinstance(history, List):
            history = []

        query = self.tokenizer.from_list_format(prompt)

        raw_text, context_tokens, history = self.make_context(query=query, history=history)

        prompt_tokens = len(context_tokens)
        if self.logger:
            self.logger.info({'raw_text': raw_text, 'prompt_tokens': prompt_tokens})

        image_flag = 0
        first_token_latency = None
        last_token_time = time.time()
        token_latency = []
        start = time.time()
        if (not use_lora) and self.is_lora:
            with self.model.disable_adapter():
                for response in self.model.chat_stream(tokenizer=self.tokenizer, query=query, history=history,
                                                       **kwargs):

                    response = response.replace(self.IMEND, '').replace(self.IMSTART, '').replace(
                        self.ENDOFTEXT, '')

                    if first_token_latency is None:
                        first_token_latency = time.time() - start
                    token_latency.append(time.time() - last_token_time)
                    token_latency.sort()
                    avg_token_latency = sum(token_latency) / len(token_latency)
                    last_token_time = time.time()
                    generation_tokens = self.token_counter(response)
                    time_cost = time.time() - start

                    if time_cost > self.dead_line:
                        break

                    average_speed = f"{generation_tokens / time_cost:.3f} token/s"
                    temp_history = deepcopy(history)
                    temp_history.append((query, response))
                    resp = {
                        "model_name": self.model_name,
                        "type": "text",
                        "answer": "",
                        "history": temp_history,
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

                    if image_flag == 1:
                        resp.update({'answer': '正在绘图中......'})
                        yield resp
                    else:
                        if self.tokenizer.ref_start_tag in response or self.tokenizer.box_start_tag in response:
                            image_flag = 1
                            resp.update({'answer': '正在绘图中......'})
                            yield resp
                        else:
                            resp.update({'answer': response, 'history': temp_history})
                            yield resp

                if image_flag == 1:
                    # image = self.tokenizer.draw_bbox_on_latest_picture(response, temp_history)
                    # image.save('xx.jpg')
                    image = Image.fromarray(
                        self.tokenizer.draw_bbox_on_latest_picture(response, temp_history).get_image())
                    resp.update({"type": "image", "image": pil_base64(image), "answer": "绘图完成"})
                    yield resp
        else:
            for response in self.model.chat_stream(tokenizer=self.tokenizer, query=query, history=history,
                                                   **kwargs):

                response = response.replace(self.IMEND, '').replace(self.IMSTART, '').replace(
                    self.ENDOFTEXT, '')
                if first_token_latency is None:
                    first_token_latency = time.time() - start
                token_latency.append(time.time() - last_token_time)
                token_latency.sort()
                avg_token_latency = sum(token_latency) / len(token_latency)
                last_token_time = time.time()
                generation_tokens = self.token_counter(response)
                time_cost = time.time() - start

                # exit
                if time_cost > self.dead_line:
                    break

                average_speed = f"{generation_tokens / time_cost:.3f} token/s"
                temp_history = deepcopy(history)
                temp_history.append((query, response))
                resp = {
                    "model_name": self.model_name,
                    "type": "text",
                    "answer": "",
                    "history": temp_history,
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

                if image_flag == 1:
                    resp.update({'answer': '正在绘图中......'})
                    yield resp
                else:
                    if self.tokenizer.ref_start_tag in response or self.tokenizer.box_start_tag in response:
                        image_flag = 1
                        resp.update({'answer': '正在绘图中......'})
                        yield resp
                    else:
                        resp.update({'answer': response, 'history': temp_history})
                        yield resp

            if image_flag == 1:
                # image = self.tokenizer.draw_bbox_on_latest_picture(response, temp_history)
                # image.save('xx.jpg')
                image = Image.fromarray(
                    self.tokenizer.draw_bbox_on_latest_picture(response, temp_history).get_image())
                resp.update({"type": "image", "image": pil_base64(image), "answer": "绘图完成"})
                yield resp

        torch_gc(self.device)
        gc.collect()
        if self.logger:
            self.logger.info(resp)
