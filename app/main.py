import base64
from typing import Optional
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from app.graph import agent
from app.llm import get_llm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MedGuide AI")
vision_llm = get_llm("VISION_MODEL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)
def describe_image(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": "Describe the visible medical-relevant features of this image "
                                 "objectively in 3-4 sentences. Do not diagnose."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
    return vision_llm.invoke([msg]).content

@app.post("/chat")
async def chat(question: str = Form(...), image: Optional[UploadFile] = File(None)):
    q = question
    if image:
        obs = describe_image(await image.read(), image.content_type or "image/png")
        q = f"{question}\n\n[Image observation: {obs}]"
    result = agent.invoke({"question": q})
    return {
    "answer": result["answer"],
    "route": result.get("route", "document"),
    "sources": list(dict.fromkeys(result["sources"])),
    }
    
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")