from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

from fastapi.responses import StreamingResponse

from vllm import LLM, SamplingParams


MODEL_NAME = "mistralai/Mistral-7B-v0.1"
engine_args = AsyncEngineArgs(model=MODEL_NAME)
engine = AsyncLLMEngine.from_engine_args(engine_args)
app = FastAPI()


class CreateCompletion(BaseModel):
    prompt: str
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

class CreateCompletionResponse(BaseModel):
    text: str

async def stream_generator(stream):
    prev_text = ""
    async for item in stream:
        text = item.outputs[0].text
        text_delta = text[len(prev_text):]
        prev_text = text
        yield text_delta

@app.post("/complete")
async def create_completion(params: CreateCompletion):
    sampling_params =  SamplingParams(
        temperature=params.temperature,
        max_tokens=params.max_tokens
    )
    prompt = params.prompt
    request_uid = random_uuid()

    stream = engine.generate(prompt, sampling_params, request_uid)
    if params.stream:
        return StreamingResponse(stream_generator(stream), media_type="text/plain")
    
    async for response in stream:
        final_response = response
    
    assert final_response is not None
    text = final_response.outputs[0].text
    return CreateCompletionResponse(text=text)
    