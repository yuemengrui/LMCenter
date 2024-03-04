# *_*coding:utf-8 *_*
# @Author : YueMengRui
import threading
import time
import json
from mylogger import logger
import requests
from configs import WORKER_HEART_BEAT_INTERVAL


def pretty_print_semaphore(semaphore):
    """Print a semaphore in better format."""
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def heart_beat_worker(obj):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        obj.send_heart_beat()


class BaseModelWorker:
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_name: str,
            limit_worker_concurrency: int,
            multimodal: bool = False,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_name
        self.limit_worker_concurrency = limit_worker_concurrency
        self.multimodal = multimodal
        self.call_ct = 0
        self.semaphore = None
        self.heart_beat_thread = None

    def init_heart_beat(self):
        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker,
            args=(self,),
            daemon=True,
        )
        self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/ai/worker_register"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
            "multimodal": self.multimodal,
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Model: {self.model_name}. "
            f"Semaphore: {pretty_print_semaphore(self.semaphore)}. "
            f"call_ct: {self.call_ct}. "
            f"worker_id: {self.worker_id}. "
        )

        url = self.controller_addr + "/ai/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except (requests.exceptions.RequestException, KeyError) as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if self.semaphore is None:
            return 0
        else:
            sempahore_value = (
                self.semaphore._value
                if self.semaphore._value is not None
                else self.limit_worker_concurrency
            )
            waiter_count = (
                0 if self.semaphore._waiters is None else len(self.semaphore._waiters)
            )
            return self.limit_worker_concurrency - sempahore_value + waiter_count

    def get_status(self):
        return {
            "model_name": self.model_name,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def generate_stream_gate(self, **kwargs):
        raise NotImplementedError

    async def generate_gate(self, **kwargs):
        async for x in self.generate_stream_gate(**kwargs):
            pass
        return json.loads(x[:-1].decode())
