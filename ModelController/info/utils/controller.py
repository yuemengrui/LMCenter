# *_*coding:utf-8 *_*
# @Author : YueMengRui
import time
import json
import requests
import threading
import numpy as np
import dataclasses
from typing import List
from enum import Enum, auto
from info import logger
from configs import *
from info.utils.response_code import RET, error_map


class DispatchMethod(Enum):
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


@dataclasses.dataclass
class WorkerInfo:
    model_name: str
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: str
    multimodal: bool


def heart_beat_controller(controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stale_workers_by_expiration()


class Controller:
    def __init__(self, dispatch_method: str = DISPATCH_METHOD):
        # Dict[str -> WorkerInfo]
        self.worker_info = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,)
        )
        self.heart_beat_thread.start()

    def register_worker(
            self,
            worker_name: str,
            check_heart_beat: bool,
            worker_status: dict,
            multimodal: bool,
    ):
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_name"],
            worker_status["speed"],
            worker_status["queue_length"],
            check_heart_beat,
            time.time(),
            multimodal,
        )

        logger.info(f"Register done: {worker_name}, {worker_status}")
        logger.info(f"all workers: {self.worker_info}")
        return True

    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/ai/worker/status", timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        return r.json()

    def remove_worker(self, worker_name: str):
        del self.worker_info[worker_name]

    def refresh_all_workers(self):
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(
                    w_name, w_info.check_heart_beat, None, w_info.multimodal
            ):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            model_names.add(w_info.model_name)

        return list(model_names)

    def list_multimodal_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            if w_info.multimodal:
                model_names.add(w_info.model_name)

        return list(model_names)

    def list_language_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            if not w_info.multimodal:
                model_names.add(w_info.model_name)

        return list(model_names)

    def get_worker_address(self, model_name: str):
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if model_name == w_info.model_name:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            if True:  # Directly return address
                pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # Check status before returning
            # while True:
            #     pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
            #     worker_name = worker_names[pt]
            #
            #     if self.get_worker_status(worker_name):
            #         break
            #     else:
            #         self.remove_worker(worker_name)
            #         worker_speeds[pt] = 0
            #         norm = np.sum(worker_speeds)
            #         if norm < 1e-4:
            #             return ""
            #         worker_speeds = worker_speeds / norm
            #         continue
            # return worker_name
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            for w_name, w_info in self.worker_info.items():
                if model_name == w_info.model_name:
                    worker_names.append(w_name)
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            if len(worker_names) == 0:
                return ""
            min_index = np.argmin(worker_qlen)
            w_name = worker_names[min_index]
            self.worker_info[w_name].queue_length += 1
            logger.info(
                f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}"
            )
            return w_name
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, worker_name: str, queue_length: int, **kwargs):
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    def remove_stale_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def handle_no_worker(self, params):
        logger.info(f"no worker: {params['model']}")
        ret = {
            "errcode": RET.SERVERERR,
            "errmsg": error_map[RET.SERVERERR]
        }
        return json.dumps(ret).encode() + b"\0"

    def handle_worker_timeout(self, worker_address):
        logger.info(f"worker timeout: {worker_address}")
        ret = {
            "errcode": RET.SERVERERR,
            "errmsg": error_map[RET.SERVERERR]
        }
        return json.dumps(ret).encode() + b"\0"

    # Let the controller act as a worker to achieve hierarchical
    # management. This can be used to connect isolated sub networks.
    def worker_api_get_status(self):
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name in self.worker_info:
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                speed += worker_status["speed"]
                queue_length += worker_status["queue_length"]

        model_names = sorted(list(model_names))
        return {
            "model_names": model_names,
            "speed": speed,
            "queue_length": queue_length,
        }

    def worker_api_generate_stream(self, params):
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            yield self.handle_no_worker(params)

        try:
            response = requests.post(
                worker_addr + "/ai/worker/generate",
                json=params,
                stream=True,
                timeout=WORKER_API_TIMEOUT,
            )
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:
            yield self.handle_worker_timeout(worker_addr)
