# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import uuid
import datetime

FASTAPI_TITLE = 'ModelWorker'
FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 24621

WORKER_ID = str(uuid.uuid4())[:8]

########################

ModelWorkerConfig = {
    "controller_addr": "http://model_controller:24620",  # controller地址
    "worker_addr": f"http://model_worker_yi_34b_server:{FASTAPI_PORT}",  # worker地址，将xxx替换为你的容器名
    "worker_id": WORKER_ID,
    "worker_type": "",  # vllm or others, worker类型，可以选择是否启用vllm
    "model_type": "Yi",  # ['Baichuan', 'ChatGLM3', 'Qwen2', 'Yi'] 支持的模型类型
    "model_path": "/workspace/Models/Yi-34B-Chat",  # 模型路径
    "lora_path": "",  # lora路径
    "model_name": "Yi_34B",  # 模型名
    "limit_worker_concurrency": 5,
    "multimodal": False,  # 是否是多模态模型
    "device": "cuda",
    "dtype": "bfloat16",  # ["float32", "float16", "bfloat16"],  # 模型精度，推荐bfloat16
    "gpu_memory_utilization": 0.9,
    "dead_line": 6 * 60
}


########################
LOG_DIR = f"logs/model-worker-{ModelWorkerConfig['model_name']}-{WORKER_ID}-{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
os.makedirs(LOG_DIR, exist_ok=True)
