"""
-- Created by: Ashok Kumar Pant
-- Created on: 9/16/21
"""

from typing import Dict, Optional, Any

from pydantic import BaseModel


class ModelPayload(BaseModel):
    inputs: Optional[Any] = None
    params: Optional[Dict] = {}


class CompletionPayload(BaseModel):
    context: str
    temp: Optional[float] = 1.0
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    length: Optional[int] = 256
    stop_sequence: Optional[str] = None
    next_line_only: Optional[bool] = False


class CompletionResponse(BaseModel):
    context: str
    text: str
    time: float
    stop_sequence: Optional[str] = None
    next_line_only: Optional[bool] = False


class QueueResponse(BaseModel):
    qid: int


class QueueRequest(BaseModel):
    qid: int


class BaseResponse(BaseModel):
    error: bool = False
    msg: Optional[str] = None
    result: Optional[Any] = None
