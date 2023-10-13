# LLMUtils
A bunch of useful utils when working with LLMs. These include:

* Hosting models via an API
* Estimating GPU memory usage
* Running a model via your terminal
* Finetuning an LLM

All this was tested on a g5.2xlarge GPU.

## API Utils
* `api/llm_api.py` - A basic way to host an LLM via a python FastAPI.
** Usage: `python api/llm_api.py --model-name <model_name>`

* `api/openai_api.py` - A replacement for the OpenAI API. You can run any vLLM model and sub it quickly for the OpenAI API.
** Usage: `python api/openai_api.py --model-name <model_name> --port <port>`
** Set `openai.api_key = "EMPTY"; openai.api_base = "http://localhost:{port}/v1"`