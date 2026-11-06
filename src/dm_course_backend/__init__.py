from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from dm_course_backend.services import train, load

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
    return train(body.k)


@app.get("/segments")
def get_segments():
    saved_weights = load()
    if saved_weights is None:
        raise HTTPException(503, "untrained_model") # Error Codes, HTTP 402 Bad Request, ErrorCode 102 # 1 is Kmeans error, and 2 is not trianed
    return saved_weights["summary"]

@app.get("/customers/{user_id}/segment")
def get_customer_segment(user_id: int):
    saved_weights = load()
    if saved_weights is None:
        raise HTTPException(503, "untrained_model")
    return {
        "user_id": user_id,
        "segment": saved_weights["assignments"].get(user_id, "unsegmented")
    } 

def main() -> None:
    uvicorn.run(
        "dm_course_backend:app",
        host="127.0.0.1",
        port=3001,
        reload=True
    )
