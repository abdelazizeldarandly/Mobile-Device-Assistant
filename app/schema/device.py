from typing import Dict, List, Literal

from pydantic import BaseModel, Field

MAX_INPUT_LENGTH = 500


class Device(BaseModel):
    brand: str
    model: str
    specs: Dict[str, str]
    release_year: int
    price_tier: Literal["budget", "mid-range", "flagship"]


# -------- extract ---------

class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)


class ExtractResponse(BaseModel):
    brand: str
    model: str
    specs: Dict[str, str]
    release_year: int
    price_tier: Literal["budget", "mid-range", "flagship"]


# --------- search -----------

class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)


class SearchResult(BaseModel):
    document: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]