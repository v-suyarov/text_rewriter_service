from fastapi import FastAPI
from pydantic import BaseModel
from rewriter import QwenRewriter
import uvicorn

app = FastAPI()
rewriter = None


class RewriteRequest(BaseModel):
    text: str
    prompt: str = ""


class RewriteResponse(BaseModel):
    rewritten: str


@app.on_event("startup")
def load_model():
    global rewriter
    rewriter = QwenRewriter()


@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite(request: RewriteRequest):
    print(f"📝 Получен запрос на рерайт: {request}")
    rewritten = rewriter.rewrite(request.text, request.prompt)
    return RewriteResponse(rewritten=rewritten)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
