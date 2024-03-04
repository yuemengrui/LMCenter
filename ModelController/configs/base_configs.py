# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os

FASTAPI_TITLE = 'ModelController'
FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 24620

########################
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

########################
CONTROLLER_HEART_BEAT_EXPIRATION = 90
WORKER_API_TIMEOUT = 120

DISPATCH_METHOD = 'shortest_queue'  # 'shortest_queue' or 'lottery'
