# Introduction
This project demonstrates using Ray serve to deploy a large language model inference pipeline. 
 The inference pipeline applies the following steps to a user prompt: 
 1. text processing to remove special characters ($, * etc) from the prompt and apply min/max number of characters constraint
 2. Tokenization to convert words into tokens from the vocabulary of the large language model
 3. Running the [large language model](https://huggingface.co/sgugger/sharded-gpt-j-6B) on the tokens to generate new tokens using the decoding scheme specified by the user
 4. Decoding the output tokens into English words

![](img/multi-model-pipeline.png)
These four steps are implemented using three models - *validator* [src\validator.py], *tokenizer* [src\tokenizer.py] and *generator* [src\generator.py]. Each of these models is a Ray deployment which are composed together into a [deployment pipeline](https://docs.ray.io/en/latest/serve/model_composition.html#serve-model-composition). 

The entrypoint to the deployment is the Run method in Generate. This method is exposed at */run* and applies the 4 steps listed above in sequence. 

While the system could be deployed anywhere with the system resources to run inference on large language models, I've tested the system on my personal computer equipped with 20 cores and 2 1080 Ti GPUs. Each GPU has about 11GB of usable RAM. At ~16GB, the language model is too big to fit on a single GPU, so I use a [new feature](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) in Huggins Face to dispatch the model parameters across available devices (such as multiple GPUs and CPUs). Using this technique, inference can happen even if the model doesn’t fit on one of the GPUs or the CPU RAM! Huggins Face inference library takes care of transferring the input to a layer to the appropriate device. 

A super cool feature of Ray deployments is that each deployment in a deployment pipeline is independently scaleable. The user can specify the min and max number of replicas for each deployment, as well as the number of ongoing requests for each replica (*target_num_ongoing_requests_per_replica*) as the autoscaling metric. If the number of requests waiting to be processed exceeds this number, the Ray autoscaler will spawn new replicas upto the max number of replicas specified by the user and given sufficient compute/memory resources on the cluster to accomodate these replicas. Conversely, if the request volume is low, replicas will be removed until the number of requests waiting to be processed starts approaching the autoscaling target. 

In this deployment pipeline, the processing time is dominated by the Generate deployment. Assume that each generate request takes 5 sec, and the other steps in the pipeline take 0.5 sec. If the expected number of requests per sec is 2 and desired max latency for serving each request is 6 sec, then we'd need two replicas of the Generate deployment for the pipeline throughput to meet the latency SLA. This can be achieved by setting *target_num_ongoing_requests_per_replica* = 1. As incoming requests pile up, the Ray autoscaler will spawn another replica of the Generate deployment. 

Because my computer only has 2 GPUs, and both GPUs are utilized to store model parameters and run inference, only one replica of the Generate deployment can be deployed at a time. This means that the system can only serve a single request at a time. This limitation arises from the lack of compute resources on my computer, and not because of a limitation in Ray.

## Rate Limitation
 Rate limiting is commonly used to improve the availability of API-based services by avoiding resource starvation. It helps maintain system availability by preventing excessive use--whether intended or unintended. 
 For microservices built using FastAPI, [Fastapi-limiter](https://github.com/long2ice/fastapi-limiter) is a popular package to add route specific rate limits. 
 
 Using Fastapi-limiter requires hooking into the startup event of the FastAPI app, as shown below
 ```sql
@app.on_event("startup")
async def startup():
    redis = redis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis)

```
This requires manually creating the FastAPI app object, rather than relying on Serve to do so internally. Fortunately, Serve provides a more explicit [integration with FastAPI](https://docs.ray.io/en/latest/serve/http-guide.html), allowing you to define a Serve deployment using the @serve.ingress decorator that wraps a FastAPI app with its full range of features. A Serve deployment can be created like this:
```python
@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 2.0},  autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 5,
        "target_num_ongoing_requests_per_replica": 10,
    })
@serve.ingress(app)
class Generate:
    @app.post("/run", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
    async def run(self, request: Request) -> JSONResponse:
        print("processing request")
        # Run inference
``` 
Note that FastAPI rate limiter requires a redis server to be running. By default, Redis listens on port 6379, which also happens to be the port Ray's GCS listens on. To avoid conflict, redis-server must be instructed to use a different port. This can be done like this:
```python
redis-server --port 6380
```

## Number of requests in queue
A text generation request takes ~5 sec to complete. Because of hardware resource limitations, only a single request can be processed at a time. This means that concurrent requests the system will be queued up and processed sequentially. To provide users an estimate of how long it will take to process their request, it is desirable to report the number of requests waiting to be processed, which is a good proxy for expected response time. 

I use Ray observability APIs to obtain a summary of tasks. I then filter this list of tasks by "handle_request" tasks that are in the "RUNNING" state. My experiments revealed this to be a good proxy for requests waiting to be served. There could be a more direct way to obtain this information as well. The code for this is in metrics.py

```python
@app.get("/metrics", dependencies=[Depends(RateLimiter(times=500, seconds=10))])
def metrics() -> JSONResponse:
    num_pending_requests = 0
    try:  # raise_on_missing_output flag prevents throwing an exception when the returned data is too large and
        # when the returned data is too large
        tasks = summarize_tasks(raise_on_missing_output=False)
    except Exception as e:
        print(e)
    else:
        summary = tasks["cluster"]["summary"]

        for k, v in summary.items():
            if "handle_request" in k:
                if v['state_counts'].get('RUNNING'):
                    num_pending_requests += v['state_counts']['RUNNING']
    finally:
        return JSONResponse(num_pending_requests)
```
As Ray continues running, the number of tasks in the system continues to increase. Thus, retrieving and filtering Ray tasks using the logic above gets progressively slower. To avoid this slowdown, Ray can be instructed to only keep a record of a certain number of tasks, and discard information about other tasks. This is done at Ray startup time using the `RAY_task_events_max_num_task_in_gcs` parameter
```python
RAY_task_events_max_num_task_in_gcs=500 ray start --head --dashboard-host=0.0.0.0 &
```
## Payload
## System deployment
### Without docker
### Using docker

Currently the system uses Contrastive Search decoding strategy during text generation. Upcoming features include making the decoding strategy and the number of tokens generated configurable by the user. 

## System Architecture
![](img/system_architecture.png)
