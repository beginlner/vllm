import os
import requests
import threading
import argparse
import asyncio
import codecs
import json
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.metrics import add_global_metrics_labels
from vllm.entrypoints.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

from .vllm_manager import VLLMManager, VLLMManagerArgs
from .vllm_generator import VLLMGenerator

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
