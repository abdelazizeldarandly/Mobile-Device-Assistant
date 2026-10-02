import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException

from app.schema.device import (
    ExtractRequest,
    ExtractResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.extraction import ExtractionError, extract_device
from app.services.search import hybrid_search
from app.utils.security import PromptInjectionError, check_prompt_injection

app = FastAPI(
    title="Mobile Device Assistant API",
    description="Structured device spec extraction and hybrid (BM25 + dense + RRF) search.",
)


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest):
    # Layer 1 of the injection defense
    try:
        check_prompt_injection(request.text)
    except PromptInjectionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        device = extract_device(request.text)
    except ExtractionError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except RuntimeError as e:
        #OPENAI Api Key missing
        raise HTTPException(status_code=500, detail=str(e))

    return ExtractResponse(**device.model_dump())


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    results = hybrid_search(request.q, k=3)
    return SearchResponse(
        results=[SearchResult(document=doc, score=score) for doc, score in results]
    )