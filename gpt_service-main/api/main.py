"""
-- Created by: Ashok Kumar Pant
-- Created on: 12/29/20
"""

import logging
import traceback

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import sys
sys.path.append('/path/to/gptservice')


from gptservice.entities import BaseResponse
from gptservice.entities import CompletionPayload, CompletionResponse, QueueRequest, QueueResponse
import gptservice.gptj_service


app = FastAPI(title="GPT Service")

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# globals
GPTJ_MODEL = None


@app.get("/")
async def hc():
    return BaseResponse(error=False, msg="OK")


@app.on_event("startup")
async def startup_event():
    global GPTJ_MODEL
    try:
        GPTJ_MODEL = get_gptj_model()
        GPTJ_MODEL.load_model()
        GPTJ_MODEL.start_background()
    except Exception as e:
        logger.debug(f"Model could not be loaded: {str(e)}")
        traceback.print_exc()


@app.post("/get_prediction")
def get_prediction(payload: QueueRequest) -> BaseResponse:
    res = GPTJ_MODEL.wait_for_queue(payload.qid)
    return BaseResponse(error=False, result=CompletionResponse(**res))


@app.post("/predict")
def model_prediction(payload: CompletionPayload) -> BaseResponse:
    res = GPTJ_MODEL.add_to_queue(payload)
    return BaseResponse(error=False, result=QueueResponse(qid=res['qid']))


@app.post("/generate")
def generate(payload: CompletionPayload) -> BaseResponse:
    res = GPTJ_MODEL.add_to_queue(payload)
    res = GPTJ_MODEL.wait_for_queue(res['qid'])
    return BaseResponse(error=False, result=CompletionResponse(**res))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
