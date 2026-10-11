import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

print("Starting server with settings:")
print("GROQ_API_KEY:", "set" if os.environ.get("GROQ_API_KEY") else "not set")
print("GROQ_MODEL:", os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"))

# 1. Read settings from environment variables (never hard-code the key)
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

client = Groq(api_key=API_KEY)
app = FastAPI()


# 2. The shape of the request the browser sends
class ChatRequest(BaseModel):
    question: str


# 3. The backend endpoint: question in, answer out
@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    try:
        result = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            max_tokens=512,
        )
        return {"answer": result.choices[0].message.content}
    except Exception as e:
        print("Groq call failed:", repr(e))  # visible in the server logs
        raise HTTPException(status_code=502, detail="The model call failed")


# 4. A simple health check (handy later for load balancers)
@app.get("/health")
def health():
    return {"status": "ok"}


# 5. Serve the chat page
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")
