import argparse
import asyncio
import os
from contextlib import asynccontextmanager

import fastapi
from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
from fastapi.middleware.cors import CORSMiddleware

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.metrics import add_global_metrics_labels
from vllm.logger import init_logger
from .vllm_generator import VLLMGenerator
from .vllm_manager import VLLMManager, VLLMManagerArgs

TIMEOUT_KEEP_ALIVE = 5  # seconds
VLLM_HEART_BEAT_INTERVAL = int(os.getenv("VLLM_WORKER_HEART_BEAT_INTERVAL", 15))
logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="deepseek generate entrypoint for vllm engine")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")

    parser = VLLMManagerArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


args = parse_args()


engine_args: AsyncEngineArgs = None
engine: AsyncLLMEngine = None
manager: VLLMManager = None
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    if engine.run_forever:
        engine.start_background_loop()

    # start manager
    manager.start()

    yield

# fastapi instance
app = fastapi.FastAPI(lifespan=lifespan)
app.add_middleware(MetricsMiddleware)  # Trace HTTP server metrics
app.add_route("/metrics", metrics)  # Exposes HTTP metrics
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# vllm engine
engine_args = AsyncEngineArgs.from_cli_args(args)
engine = AsyncLLMEngine.from_engine_args(engine_args)
engine_model_config = asyncio.run(engine.get_model_config())
max_model_len = engine_model_config.max_model_len
# Register labels for metrics
add_global_metrics_labels(model_name=engine_args.model)
# add generate router
app.include_router(VLLMGenerator(engine=engine), prefix="/v1")
# create vllm manager
manager_args = VLLMManagerArgs.from_cli_args(args)
manager = VLLMManager.from_manager_args(manager_args)
