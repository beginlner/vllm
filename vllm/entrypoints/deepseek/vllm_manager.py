import argparse
import dataclasses
import os
import threading
import time

import requests

from vllm.logger import init_logger

VLLM_HEART_BEAT_INTERVAL = int(os.getenv("VLLM_WORKER_HEART_BEAT_INTERVAL", 15))
logger = init_logger(__name__)


@dataclasses.dataclass
class VLLMManagerArgs(object):
    """
    worker_addr: address of this entrypoint of ingress, like ' http://10.2.208.251:80/system01/{TASK_NAME}'
    controller_addr: deepseek api server LB address, like 'http://172.30.4.182:1090/loadbalance'
    """
    worker_address: str = "http://localhost:8000"
    controller_address: str = "http://localhost:21001"
    register: bool = False
    served_model_name: str = "vllm_model"

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'VLLMManagerArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--worker-address",
                            type=str,
                            default="http://localhost:8000")
        parser.add_argument("--controller-address",
                            type=str,
                            default="http://localhost:21001")
        parser.add_argument("--register",
                            action="store_true",
                            help="Whether to regitser to the controller.")
        parser.add_argument("--served-model-name",
                            type=str,
                            default=None,
                            help="The model name used in the API. If not "
                                 "specified, the model name will be the same as "
                                 "the huggingface name.")
        return parser


class VLLMManager:
    @classmethod
    def from_manager_args(cls, manager_args: VLLMManagerArgs) -> 'VLLMManager':
        manager = cls(**manager_args.__dict__)
        return manager

    def __init__(
        self,
        worker_address: str,
        controller_address: str,
        served_model_name: str,
        register: bool,
    ):
        self.worker_addr = worker_address
        self.controller_addr = controller_address
        self.model = served_model_name
        self.register = register
        self.headers = {"Host": "coder.deepseek.com"}

        self.heart_beat_thread = None
        if register:
            self.running = True
            self.init_heart_beat()

        self.suspend_thread = None
        if self.register:
            self.init_suspend_thread()

    def start(self):
        logger.info('start vllm manager threads')
        if self.register:
            self.heart_beat_thread.start()
            self.suspend_thread.start()

    def init_heart_beat(self):

        def heart_beat_worker():
            time.sleep(5)  # Waiting server for initialization.
            self._register_to_controller()
            while True:
                time.sleep(VLLM_HEART_BEAT_INTERVAL)
                self._send_heart_beat()

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker,
            daemon=True,
        )
        # self.heart_beat_thread.start()

    def _register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_addr": self.worker_addr,
            "model": self.model,
        }
        while True:
            try:
                ret = requests.post(url,
                                    json=data,
                                    timeout=5,
                                    headers=self.headers)
                status = ret.json()["code"]
                break
            except (requests.exceptions.RequestException, KeyError) as e:
                logger.error(f"register worker error: {e}")
            time.sleep(5)

        assert status == 0

    def _send_heart_beat(self):
        logger.info(f"Send heart beat. Model: {self.model} ")

        url = self.controller_addr + "/receive_heart_beat"
        data = {
            "worker_addr": self.worker_addr,
            "model": self.model,
        }
        while True:
            try:
                ret = requests.post(url,
                                    json=data,
                                    timeout=5,
                                    headers=self.headers)
                status = ret.json()["code"]
                break
            except (requests.exceptions.RequestException, KeyError) as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if status != 0 and self.running:
            self._register_to_controller()

    def init_suspend_thread(self):
        def suspend_worker():
            import hfai
            while True:
                if hfai.receive_suspend_command():
                    logger.info(f"Receive suspend command")
                    break
                time.sleep(1)
            self.running = False
            self._send_suspend()
            logger.info(f"Suspend command sended to controller")

        self.suspend_thread = threading.Thread(
            target=suspend_worker,
            daemon=True,
        )
        # self.suspend_thread.start()

    def _send_suspend(self):
        logger.info("Send suspend to controller")

        url = self.controller_addr + "/receive_suspend"
        data = {
            "worker_addr": self.worker_addr,
            "model": self.model,
        }
        while True:
            try:
                ret = requests.post(url,
                                    json=data,
                                    timeout=5,
                                    headers=self.headers)
                status = ret.json()["code"]
                break
            except (requests.exceptions.RequestException, KeyError) as e:
                logger.error(f"send suspend error: {e}")
            time.sleep(5)

        assert status == 0

