# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
import asyncio
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

import fastapi
import uvicorn
import os
import requests
import threading
from fastapi import Request, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from packaging import version

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
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

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template
    from fastchat.conversation import get_conv_template
    _fastchat_available = True
except ImportError:
    _fastchat_available = False

TIMEOUT_KEEP_ALIVE = 5  # seconds
VLLM_HEART_BEAT_INTERVAL = int(os.getenv("VLLM_WORKER_HEART_BEAT_INTERVAL",
                                         15))

logger = init_logger("vllm")
served_model = None
app = fastapi.FastAPI()
engine = None
num_unfinished_requests = 0


def increase():
    global num_unfinished_requests
    num_unfinished_requests += 1


def decrease():
    global num_unfinished_requests
    num_unfinished_requests -= 1


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(decrease)
    return background_tasks


class VLLMManager:

    def __init__(
        self,
        worker_addr: str,
        controller_addr: str,
        model: str,
        register: bool,
    ):
        self.worker_addr = worker_addr
        self.controller_addr = controller_addr
        self.model = model
        self.register = register
        self.headers = {"Host": "coder.deepseek.com"}

        self.heart_beat_thread = None
        if register:
            self.running = True
            self.init_heart_beat()

        self.suspend_thread = None
        self.init_suspend_thread()

    def init_heart_beat(self):

        def heart_beat_worker():
            time.sleep(5)  # Waiting server for initialization.
            self.register_to_controller()
            while True:
                time.sleep(VLLM_HEART_BEAT_INTERVAL)
                self.send_heart_beat()

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker,
            daemon=True,
        )
        self.heart_beat_thread.start()

    def register_to_controller(self):
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

    def send_heart_beat(self):
        global num_unfinished_requests
        logger.info(f"Send heart beat. Model: {self.model} "
                    f"num_unfinished_requests: {num_unfinished_requests}")

        url = self.controller_addr + "/receive_heart_beat"
        data = {
            "worker_addr": self.worker_addr,
            "model": self.model,
            "num_unfinished_requests": num_unfinished_requests,
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
            self.register_to_controller()

    def init_suspend_thread(self):

        def suspend_worker():
            import hfai
            while True:
                if hfai.receive_suspend_command():
                    logger.info(f"Receive suspend command")
                    break
                time.sleep(1)
            if self.register:
                self.running = False
                self.send_suspend()
            time.sleep(5)  # To be conservative.
            global num_unfinished_requests
            while num_unfinished_requests > 0:
                logger.info(
                    f"num_unfinished_requests: {num_unfinished_requests}")
                time.sleep(1)
            time.sleep(3)  # To be conservative.
            logger.info(f"Go suspend")
            os.environ["MARSV2_TASK_TYPE"] = "training"  # Hack
            hfai.go_suspend()

        self.suspend_thread = threading.Thread(
            target=suspend_worker,
            daemon=True,
        )
        self.suspend_thread.start()

    def send_suspend(self):
        if not self.register:
            return

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


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


def _add_to_set(s, new_stop) -> None:
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


def _merge(x, y) -> List:
    ret = set()
    _add_to_set(x, ret)
    _add_to_set(y, ret)
    return list(ret)


async def get_gen_params(request: ChatCompletionRequest) -> Tuple[str, Dict]:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install fschat`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install -U fschat`")

    if conv_template:
        conv = get_conv_template(conv_template)
    else:
        conv = get_conversation_template(request.model)
        conv = Conversation(
            name=conv.name,
            system_template=conv.system_template,
            system_message=conv.system_message,
            roles=conv.roles,
            messages=list(conv.messages),  # prevent in-place modification
            offset=conv.offset,
            sep_style=SeparatorStyle(conv.sep_style),
            sep=conv.sep,
            sep2=conv.sep2,
            stop_str=conv.stop_str,
            stop_token_ids=conv.stop_token_ids,
        )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    sampling_params = dict(
        n=request.n,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=_merge(request.stop, conv.stop_str),
        stop_token_ids=_merge(request.stop_token_ids, conv.stop_token_ids),
        max_tokens=request.max_tokens,
        best_of=request.best_of,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
        use_beam_search=request.use_beam_search,
        skip_special_tokens=request.skip_special_tokens,
        spaces_between_special_tokens=request.spaces_between_special_tokens,
    )

    return prompt, sampling_params


async def check_length(
    request: Union[ChatCompletionRequest, CompletionRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    input_ids = prompt_ids if prompt_ids is not None else tokenizer(
        prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num >= max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {token_num} in the messages. "
            f"Please reduce the length of the messages.",
        )
    else:
        request.max_tokens = min(request.max_tokens, max_model_len - token_num)
        return input_ids, None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model,
                  root=served_model,
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


def create_logprobs(
    token_ids: List[int],
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None,
    num_output_top_logprobs: Optional[int] = None,
    initial_text_offset: int = 0,
) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    if num_output_top_logprobs:
        logprobs.top_logprobs = []
    for i, token_id in enumerate(token_ids):
        step_top_logprobs = top_logprobs[i]
        if step_top_logprobs is not None:
            token_logprob = step_top_logprobs[token_id]
        else:
            token_logprob = None
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(token_logprob)
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] +
                                        last_token_len)
        last_token_len = len(token)

        if num_output_top_logprobs:
            logprobs.top_logprobs.append({
                tokenizer.convert_ids_to_tokens(i): p
                for i, p in step_top_logprobs.items()
            } if step_top_logprobs else None)
    return logprobs


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    # logger.info(f"Received chat completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    prompt, sampling_params = await get_gen_params(request)
    token_ids, error_check_ret = await check_length(request, prompt=prompt)
    sampling_params["max_tokens"] = request.max_tokens
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = request.id or f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    try:
        sampling_params = SamplingParams(**sampling_params)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt, sampling_params, request_id,
                                       token_ids)

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
        usage: Optional[UsageInfo] = None,
        payload: Optional[Tuple[float, int]] = None,
        last_request_id: Optional[str] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(content=text),
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
            payload=payload,
            last_request_id=last_request_id,
        )
        if usage is not None:
            response.usage = usage
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 choices=[choice_data],
                                                 model=model_name,
                                                 payload=None,
                                                 last_request_id=None)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                completion_tokens = len(output.token_ids)
                previous_num_tokens[i] = completion_tokens
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                    payload=res.payload,
                    last_request_id=res.last_request_id,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    prompt_tokens = len(res.prompt_token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        finish_reason=output.finish_reason,
                        usage=final_usage,
                        payload=res.payload,
                        last_request_id=res.last_request_id,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        increase()
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream",
                                 background=create_background_tasks())

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=ChatMessage(role="assistant", content=output.text),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
        payload=final_res.payload,
        last_request_id=final_res.last_request_id,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        increase()
        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream",
                                 background=create_background_tasks())

    return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by vLLM engine)
    """
    # logger.info(f"Received completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    # OpenAI API supports echoing the prompt when max_tokens is 0.
    echo_without_generation = request.echo and request.max_tokens == 0

    if request.suffix is not None:
        # The language models we currently support do not support suffix.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "suffix is not currently supported")

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    model_name = request.model
    request_id = request.id or f"cmpl-{random_uuid()}"

    use_token_ids = False
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "please provide at least one prompt")
        first_element = request.prompt[0]
        if isinstance(first_element, int):
            use_token_ids = True
            prompt = request.prompt
        elif isinstance(first_element, (str, list)):
            # TODO: handles multiple prompt case in list[list[int]]
            if len(request.prompt) > 1:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    "multiple prompts in a batch is not currently supported")
            use_token_ids = not isinstance(first_element, str)
            prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if use_token_ids:
        _, error_check_ret = await check_length(request, prompt_ids=prompt)
    else:
        token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    created_time = int(time.monotonic())
    try:
        spaces_between_special_tokens = request.spaces_between_special_tokens
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_tokens
            if not echo_without_generation else 1,
            logprobs=request.logprobs,
            use_beam_search=request.use_beam_search,
            prompt_logprobs=request.logprobs if request.echo else None,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    if use_token_ids:
        result_generator = engine.generate(None,
                                           sampling_params,
                                           request_id,
                                           prompt_token_ids=prompt)
    else:
        result_generator = engine.generate(prompt, sampling_params, request_id,
                                           token_ids)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    stream = (request.stream
              and (request.best_of is None or request.n == request.best_of)
              and not request.use_beam_search)

    def create_stream_response_json(
        index: int,
        text: str,
        logprobs: Optional[LogProbs] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[UsageInfo] = None,
        payload: Optional[Tuple[float, int]] = None,
        last_request_id: Optional[str] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
            payload=payload,
            last_request_id=last_request_id,
        )
        if usage is not None:
            response.usage = usage
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        has_echoed = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                token_ids = output.token_ids[previous_num_tokens[i]:]
                top_logprobs = output.logprobs[previous_num_tokens[i]:]
                offsets = len(previous_texts[i])
                if request.echo and not has_echoed[i]:
                    if not echo_without_generation:
                        delta_text = res.prompt + delta_text
                        token_ids = res.prompt_token_ids + token_ids
                        top_logprobs = res.prompt_logprobs + top_logprobs
                    else:
                        delta_text = res.prompt
                        token_ids = res.prompt_token_ids
                        top_logprobs = res.prompt_logprobs
                    has_echoed[i] = True
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        token_ids=token_ids,
                        top_logprobs=top_logprobs,
                        num_output_top_logprobs=request.logprobs,
                        initial_text_offset=offsets,
                    )
                else:
                    logprobs = None
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                finish_reason = output.finish_reason
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                    logprobs=logprobs,
                    finish_reason=finish_reason,
                    payload=res.payload,
                    last_request_id=res.last_request_id,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    logprobs = (LogProbs()
                                if request.logprobs is not None else None)
                    prompt_tokens = len(res.prompt_token_ids)
                    completion_tokens = len(output.token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                        usage=final_usage,
                        payload=res.payload,
                        last_request_id=res.last_request_id,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if stream:
        increase()
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream",
                                 background=create_background_tasks())

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    prompt_token_ids = final_res.prompt_token_ids
    prompt_logprobs = final_res.prompt_logprobs
    prompt_text = final_res.prompt
    for output in final_res.outputs:
        if request.logprobs is not None:
            if not echo_without_generation:
                token_ids = output.token_ids
                top_logprobs = output.logprobs
                if request.echo:
                    token_ids = prompt_token_ids + token_ids
                    top_logprobs = prompt_logprobs + top_logprobs
            else:
                token_ids = prompt_token_ids
                top_logprobs = prompt_logprobs
            logprobs = create_logprobs(
                token_ids=token_ids,
                top_logprobs=top_logprobs,
                num_output_top_logprobs=request.logprobs,
            )
        else:
            logprobs = None
        if not echo_without_generation:
            output_text = output.text
            if request.echo:
                output_text = prompt_text + output_text
        else:
            output_text = prompt_text
        choice_data = CompletionResponseChoice(
            index=output.index,
            text=output_text,
            logprobs=logprobs,
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
        payload=final_res.payload,
        last_request_id=final_res.last_request_id,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        increase()
        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream",
                                 background=create_background_tasks())

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--worker-address",
                        type=str,
                        default="http://localhost:8000")
    parser.add_argument("--controller-address",
                        type=str,
                        default="http://localhost:21001")
    parser.add_argument("--register",
                        action="store_true",
                        help="Whether to regitser to the controller.")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")
    parser.add_argument("--conv-template",
                        type=str,
                        default=None,
                        help="The conversation template name.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model
    conv_template = args.conv_template

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.max_model_len

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_model_config.tokenizer,
        tokenizer_mode=engine_model_config.tokenizer_mode,
        trust_remote_code=engine_model_config.trust_remote_code)

    vllm_manager = VLLMManager(
        args.worker_address,
        args.controller_address,
        served_model,
        args.register,
    )

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
