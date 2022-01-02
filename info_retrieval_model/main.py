from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core import VectorialModel, initialize

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vm = initialize()

def prepare_output(documents):
    body = []
    for doc in documents:
        body.append({
            "ranking": doc[0],
            "text": doc[1].text[:min(len(doc[1].text), 400)] + ("(...)" if len(doc[1].text) > 400 else ""),
            "id": doc[2]

        })
    return body

@app.get("/query")
def query_docs(value: str = ""):
    documents = vm.query(value)
    return prepare_output(documents)

@app.get("/document/{doc_id}")
def read_document(doc_id: int):
    return { "text": vm.docs[doc_id].text }

