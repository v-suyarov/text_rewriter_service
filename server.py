from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from rewriter import QwenRewriter
from tag_predictor import QwenTagPredictor
import uvicorn

app = FastAPI()
rewriter = None
tag_predictor = None


class RewriteRequest(BaseModel):
    text: str
    prompt: str = ""


class RewriteResponse(BaseModel):
    rewritten: str


class TagRequest(BaseModel):
    text: str
    available_tags: List[str]


class TagResponse(BaseModel):
    tags: List[str]


@app.on_event("startup")
def load_models():
    global rewriter, tag_predictor
    rewriter = QwenRewriter()
    tag_predictor = QwenTagPredictor()


@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite(request: RewriteRequest):
    print(f"📝 Получен запрос на рерайт: {request}")
    rewritten = rewriter.rewrite(request.text, request.prompt)
    return RewriteResponse(rewritten=rewritten)


@app.post("/predict_tags", response_model=TagResponse)
async def predict_tags(request: TagRequest):
    print(f"📝 Получен запрос на определение тегов: {request}")
    print(request.available_tags)
    tags = tag_predictor.predict_tags(request.text, request.available_tags)
    print(f"📝 Определены теги: {tags}")
    return TagResponse(tags=tags)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
