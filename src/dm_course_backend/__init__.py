from fastapi import FastAPI
import uvicorn

app = FastAPI(title="DM Backend")

@app.get("/health")
def health():
    return {
        "health": "ok"
    }

def main() -> None:
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=3001,
        reload=True
    )
