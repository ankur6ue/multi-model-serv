
import redis.asyncio as redis
from starlette.requests import Request
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from ray import serve
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import torch
import os
import time
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, BatchEncoding
from accelerate import load_checkpoint_and_dispatch
from accelerate import infer_auto_device_map
from fastapi.middleware.cors import CORSMiddleware
import logging
import ray
from tokenizer import Tokenize
from validator import Validator

# We'll use our own FASTAPI app because we need it for setting up rate limitation
app = FastAPI()
logging.basicConfig(level=logging.INFO)

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Connect to Redis for rate limitation
@app.on_event("startup")
async def startup():
    redis_ = redis.from_url("redis://localhost:6380", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_)
    print('Initialized FastAPILimiter')


@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 2.0},  autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 5,
        "target_num_ongoing_requests_per_replica": 10,
    })
@serve.ingress(app)
class Generate:
    def __init__(self, tokenizer, validator):
        self.tokenizer = tokenizer
        self.validator = validator
        self.model = None
        # MODEL_PATH key is provided as a runtime environment when serve run is called.
        model_path = os.environ.get('MODEL_PATH', False)
        print("model_path: " + model_path)
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        # Load model
        checkpoint = self.model_path
        config = AutoConfig.from_pretrained(checkpoint)
        print("Loading large language model..")

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

        device_map = infer_auto_device_map(model, max_memory={0: "8GiB", 1: "10GiB", "cpu": "30GiB"},
                                           no_split_module_classes=["GPTJBlock"])

        self.model = load_checkpoint_and_dispatch(
            model, checkpoint, device_map=device_map
        )

        print("Finished loading large language model..")

    def complete(self, tokens: BatchEncoding, num_tokens:int=50, dec_strategy:str='contrastive') -> list:

        inputs = tokens.to(0)
        # contrastive search, other options are beam search, multinomial search and combinations. It is slower
        # than beam search, but demonstrates superior results for generating non-repetitive yet coherent long outputs
        if dec_strategy == 'contrastive':
            output = self.model.generate(inputs["input_ids"], penalty_alpha=0.6, top_k=4, max_new_tokens=num_tokens)
        if dec_strategy == 'multinomial sampling':
            output = self.model.generate(inputs["input_ids"], do_sample=True,  max_new_tokens=num_tokens)
        if dec_strategy == 'beam search':
            output = self.model.generate(inputs["input_ids"], num_beams=4,  max_new_tokens=num_tokens)

        return output[0].tolist()

#    async def __call__(self, http_request: Request) -> str:
    @app.post("/run", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
    async def run(self, request: Request) -> JSONResponse:
        print("processing request")
        # Run inference
        if self.model is None:
            self.load_model()

        req_json = await request.json()
        # 1. The user prompt the llm will use to generate text
        prompt: str = req_json['data']
        # 2. The number of tokens the llm will be asked to generate
        num_tokens_ = req_json.get('num_tokens')
        num_tokens: int = 50 if num_tokens_ is None else num_tokens_
        # 3. The decoding strategy: contrastive, multinomial, beam search
        dec_strat_ = req_json.get('decoding_strategy')
        dec_strat: str = 'contrastive' if dec_strat_ is None else dec_strat_
        print("prompt: {0}, num_tokens: {1}, decoding strategy: {2}".format(prompt, num_tokens, dec_strat))
        # check if the prompt contains any unallowed characters or exceeds character bound
        is_valid_ref = await self.validator.validate.remote(prompt)
        is_valid = await is_valid_ref
        if not is_valid:
            raise HTTPException(status_code=400, detail="Invalid prompt")
        encoder_ref = await self.tokenizer.encode.remote(prompt)
        tokens = await encoder_ref
        completed_tokens = self.complete(tokens, num_tokens, dec_strat)
        decoder_ref = await self.tokenizer.decode.remote(completed_tokens)
        decoded_text = await decoder_ref
        print(decoded_text)
        return JSONResponse(decoded_text)


# ray.init(num_cpus=10, num_gpus=2, dashboard_host="127.0.0.1")
generate = Generate.bind(Tokenize.bind(), Validator.bind())
# To run this from the Pycharm debugger, uncomment the line below, and set run_in_debugger = 1.
# Also set the value of model_path to the directory where the generate text model is stored
# ray.serve.run(target=generate, host="127.0.0.1")
run_in_debugger = 0
if run_in_debugger:
    while 1:
        time.sleep(100)
# ray start --head --dashboard-host=0.0.0.0
# ray.serve.start(target=autocomplete, http_options={"host": "0.0.0.0"})

