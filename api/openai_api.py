# A FastAPI server that stands in for the OpenAI API but uses the vLLM engine.
import argparse
import asyncio
import json
import time
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, Request
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid
import logging

from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    LogProbs,
    UsageInfo,
)

from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

app = FastAPI()
engine = None
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return JSONResponse(
        status_code=HTTPStatus.BAD_REQUEST, content={"message": str(exc)}
    )


async def generate_prompt_from_messages(
    model: str, messages: List[Dict[str, str]]
) -> str:
    conv = get_conversation_template(model)
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
    for message in messages:
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
    return conv.get_prompt()


async def check_length(
    max_tokens: Optional[int] = None,
    prompt: Optional[str] = None,
) -> Tuple[List[int], Optional[JSONResponse]]:
    input_ids = tokenizer(prompt).input_ids
    token_num = len(input_ids)

    if max_tokens is None:
        max_tokens = max_model_len - token_num
    if token_num + max_tokens > max_model_len:
        return input_ids, JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={
                "message": f"This model's maximum context length is {max_model_len} tokens. "
                f"However, you requested {max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion."
            },
        )
    else:
        return input_ids, None


def create_logprobs(
    token_ids: List[int],
    id_logprobs: List[Dict[int, float]],
    initial_text_offset: int = 0,
) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    for token_id, id_logprob in zip(token_ids, id_logprobs):
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(id_logprob[token_id])
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
        last_token_len = len(token)

        logprobs.top_logprobs.append(
            {tokenizer.convert_ids_to_tokens(i): p for i, p in id_logprob.items()}
        )
    return logprobs


def create_sampling_params(
    request: ChatCompletionRequest,
) -> Union[SamplingParams, JSONResponse]:
    try:
        return SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
        )
    except ValueError as e:
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST, content={"message": str(e)}
        )


async def stream_response_generator(
    request_id: str,
    model_name: str,
    result_gen: Any,
    logprobs: Optional[int] = None,
    chat: bool = False,
) -> AsyncGenerator[str, None]:
    prev_text = ""
    idx = 0
    async for item in result_gen:
        output = item.outputs[0]
        text = output.text
        text_delta = text[len(prev_text) :]
        if chat:
            if idx == 0:
                delta_msg = DeltaMessage(role="assistant", content=text_delta)
            else:
                delta_msg = DeltaMessage(content=text_delta)
            response = ChatCompletionStreamResponse(
                id=request_id,
                model=model_name,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=output.index,
                        delta=delta_msg,
                        finish_reason=output.finish_reason,
                    )
                ],
            )
        else:
            if logprobs is not None:
                logprobs = create_logprobs(output.token_ids, output.logprobs)
            else:
                logprobs = None
            response = CompletionStreamResponse(
                id=request_id,
                model=model_name,
                choices=[
                    CompletionResponseStreamChoice(
                        index=output.index,
                        text=text_delta,
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                    )
                ],
            )
        prev_text = text
        idx += 1
        yield response.json(ensure_ascii=False)


async def get_token_usage(
    prompt_token_ids: List[int], token_ids: List[int]
) -> UsageInfo:
    num_prompt_tokens = len(prompt_token_ids)
    num_generated_tokens = len(token_ids)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    return usage


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    logger.info(f"Received chat completion request: {request}")
    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())

    prompt = await generate_prompt_from_messages(model_name, request.messages)
    token_ids, error_check_ret = await check_length(request.max_tokens, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    sampling_params = create_sampling_params(request)
    if isinstance(sampling_params, JSONResponse):
        return sampling_params

    result_gen = engine.generate(prompt, sampling_params, request_id, token_ids)

    # Streaming response
    if request.stream:
        return StreamingResponse(
            stream_response_generator(
                request_id, model_name, result_gen, request.logprobs, chat=True
            ),
            media_type="application/json",
        )

    # Non-streaming response
    async for res in result_gen:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"message": "Client disconnected"},
            )
        final_res = res

    assert final_res is not None
    output = final_res.outputs[0]
    choice = ChatCompletionResponseChoice(
        index=output.index,
        message=ChatMessage(role="assistant", content=output.text),
        finish_reason=output.finish_reason,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=[choice],
        usage=await get_token_usage(final_res.prompt_token_ids, output.token_ids),
    )
    return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    logger.info(f"Received completion request: {request}")
    model_name = request.model
    prompt = request.prompt
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())

    # Checking prompt length
    token_ids, error_check_ret = await check_length(request.max_tokens, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    # Generating result stream
    sampling_params = create_sampling_params(request)
    if isinstance(sampling_params, JSONResponse):
        return sampling_params
    result_gen = engine.generate(prompt, sampling_params, request_id, token_ids)

    # Streaming response
    if request.stream:
        return StreamingResponse(
            stream_response_generator(
                request_id, model_name, result_gen, request.logprobs
            ),
            media_type="application/json",
        )

    # Non-streaming response
    async for res in result_gen:
        if await raw_request.is_disconnected():
            await engine.abort(request_id)
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"message": "Client disconnected"},
            )
        final_res = res
    assert final_res is not None

    output = final_res.outputs[0]
    logprobs = (
        create_logprobs(output.token_ids, output.logprobs)
        if request.logprobs is not None
        else None
    )
    choice = CompletionResponseChoice(
        index=output.index,
        text=output.text,
        logprobs=logprobs,
        finish_reason=output.finish_reason,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=[choice],
        usage=await get_token_usage(final_res.prompt_token_ids, output.token_ids),
    )
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument("--port", type=int, default=8000, help="port to run on")
    parser.add_argument(
        "--model-name", default="mistralai/Mistral-7B-v0.1", help="Model name to use"
    )

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
    engine_args = AsyncEngineArgs(model=args.model_name)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.max_model_len

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=engine_args.trust_remote_code,
    )

    uvicorn.run(app, port=args.port)
