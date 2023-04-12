from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from ray.experimental.state.api import summarize_tasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
import ray
import uvicorn


app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    redis_ = redis.from_url("redis://localhost:6380", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_)


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
