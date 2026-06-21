from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import responder

app = FastAPI(title="ChatBot Agro - General Madariaga")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pregunta(BaseModel):
    texto: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat(pregunta: Pregunta):
    if not pregunta.texto.strip():
        return {"respuesta": "Escribí una pregunta.", "fuentes": [], "modo": "error"}
    return responder(pregunta.texto)