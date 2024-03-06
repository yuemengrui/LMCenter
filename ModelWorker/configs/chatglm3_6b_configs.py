# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import uuid

FASTAPI_TITLE = 'ModelWorker'
FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 24621

WORKER_ID = str(uuid.uuid4())[:8]

########################

ModelWorkerConfig = {
    "controller_addr": "http://model_controller:24620",
    "worker_addr": f"http://model_worker_chatglm3_6b_32k_server:{FASTAPI_PORT}",
    "worker_id": WORKER_ID,
    "worker_type": "",  # vllm or others
    "model_type": "ChatGLM3",  # ['Baichuan', 'ChatGLM3', 'Qwen2']
    "model_path": "/workspace/Models/chatglm3_6b_32k",
    "model_name": "ChatGLM3_6B_32k",
    "limit_worker_concurrency": 5,
    "multimodal": False,
    "device": "cuda",
    "dtype": "bfloat16",  # ["float32", "float16", "bfloat16"],
    "gpu_memory_utilization": 0.9
}


########################
LOG_DIR = f"logs/model-worker-{ModelWorkerConfig['model_name']}-{WORKER_ID}"
os.makedirs(LOG_DIR, exist_ok=True)
