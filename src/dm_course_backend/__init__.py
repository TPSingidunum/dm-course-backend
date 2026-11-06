from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from dm_course_backend.services import train

app = FastAPI(title="DM Backend")

@app.get("/health")
def health():
    return {
        "health": "ok"
    }

class TrainRequest(BaseModel):
    k: int = Field(4, ge=2, le=10)

@app.post("/segments/train")
def train_segments(body: TrainRequest = TrainRequest()):
    print(body.k)
    return train(body.k)

def main() -> None:
    uvicorn.run(
        "dm_course_backend:app",
        host="127.0.0.1",
        port=3001,
        reload=True
    )
