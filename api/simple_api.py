from typing import Optional, Any
from fastapi import FastAPI
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

from fastapi.responses import StreamingResponse
import uvicorn
from vllm import SamplingParams
import argparse
import logging


app = FastAPI()
engine = None
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CreateCompletion(BaseModel):
    prompt: str
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False


class CreateCompletionResponse(BaseModel):
    id: Optional[str] = None
    text: str


async def stream_generator(request_id: str, stream: Any):
    logger.info(f"Starting stream generator for request_id: {request_id}")
    prev_text = ""
    async for item in stream:
        text = item.outputs[0].text
        text_delta = text[len(prev_text) :]
        prev_text = text
        yield text_delta


@app.post("/complete")
async def create_completion(params: CreateCompletion):
    logger.info(f"Received completion request: {params}")
    sampling_params = SamplingParams(
        temperature=params.temperature, max_tokens=params.max_tokens
    )
    prompt = params.prompt
    request_uid = random_uuid()
    logger.info(f"Generated request_uid: {request_uid}")
    stream = engine.generate(prompt, sampling_params, request_uid)
    if params.stream:
        logger.info("Streaming response")
        return StreamingResponse(
            stream_generator(request_uid, stream), media_type="application/json"
        )

    async for response in stream:
        final_response = response

    assert final_response is not None
    text = final_response.outputs[0].text
    logger.info(f"Final response text: {text}")
    return CreateCompletionResponse(id=request_uid, text=text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="mistralai/Mistral-7B-v0.1", help="Model name to use"
    )
    # TODO: add a model path value for custom model
    args = parser.parse_args()
    logger.info(f"Arguments parsed: {args}")

    engine_args = AsyncEngineArgs(model=args.model_name)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("Engine initialized")
    uvicorn.run(app, port=8000)
    logger.info("Server started at port 8000")
