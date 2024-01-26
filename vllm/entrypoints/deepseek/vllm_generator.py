from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from .vllm_generator_schema import VLLMGenerationSchema, \
    VLLMGenerationResponseSchema

logger = init_logger(__name__)


class VLLMGenerator(APIRouter):
    def __init__(self, engine: AsyncLLMEngine):
        super().__init__()
        self.engine = engine
        self.add_api_route('/generate', self._generate, methods=['POST'])

    async def _generate(self, vllm_generation_schema: VLLMGenerationSchema):
        # todo abort request if connect lose
        logger.info(vllm_generation_schema)
        result_generator = self.engine.generate(
            None, vllm_generation_schema.sampling_params.vllm_sampling_params(),
            vllm_generation_schema.request_id, vllm_generation_schema.token_ids)

        async def stream_generator():
            async for res in result_generator:
                res: RequestOutput
                yield VLLMGenerationResponseSchema.from_vllm_request_output(res).json_str()

        return StreamingResponse(stream_generator(), media_type='text/event-stream')
