import orjson
from typing import List, Optional, Tuple

from vllm.logger import init_logger
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import SampleLogprobs
from vllm.utils import random_uuid

logger = init_logger(__name__)
from pydantic import BaseModel


class SamplingParamsSchema(BaseModel):
    """
    Schema for SamplingParams,
    See https://platform.openai.com/docs/api-reference/chat/create
    """
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    min_p: float = 0.0
    # stop: Optional[Union[str, List[str]]] = None  # 在迭代一 vllm engine 不做 decode，所以这个 stop 的 string 就没用了
    stop_token_ids: Optional[List[int]] = None
    max_tokens: int = 16
    best_of: Optional[int] = None
    top_k: int = -1
    use_beam_search: bool = False  # VLLM 自带的

    def vllm_sampling_params(self) -> SamplingParams:
        return SamplingParams(**self.__dict__)


class VLLMGenerationSchema(BaseModel):
    sampling_params: SamplingParamsSchema = SamplingParamsSchema()
    request_id: str = f'cmp-{random_uuid()}'
    token_ids: List[int]


class VLLMGenerationOutputSchema(BaseModel):
    index: int
    token_ids: List[int]
    cumulative_logprob: float
    logprobs: Optional[SampleLogprobs] = None
    finish_reason: Optional[str] = None

    @classmethod
    def from_vllm_completion(cls, vllm_completion: CompletionOutput) -> 'VLLMGenerationOutputSchema':
        return cls(
            index=vllm_completion.index,
            token_ids=vllm_completion.token_ids,
            cumulative_logprob=vllm_completion.cumulative_logprob,
            logprobs=vllm_completion.logprobs,
            finish_reason=vllm_completion.finish_reason,
        )


class VLLMGenerationResponseSchema(BaseModel):
    request_id: str
    outputs: List[VLLMGenerationOutputSchema]
    payload: Optional[Tuple[float, int]] = None
    last_request_id: Optional[str] = None

    @classmethod
    def from_vllm_request_output(cls, request_output: RequestOutput) -> 'VLLMGenerationResponseSchema':
        return cls(request_id=request_output.request_id,
                   outputs=[VLLMGenerationOutputSchema.from_vllm_completion(o)
                            for o in request_output.outputs],
                   payload=request_output.payload,
                   last_request_id=request_output.last_request_id,
                   )

    def json_str(self):
        return orjson.dumps(self.dict(exclude_none=False))
