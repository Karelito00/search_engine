from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core import VectorialModel

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

d1 = "leon leon leon"
d2 = "leon leon leon zorro"
d3 = "leon zorro nutria"
d4 = "leon leon leon zorro zorro zorro"
d5 = "nutria"

vm = VectorialModel([d1, d2, d3, d4, d5])

def prepare_output(documents):
    body = []
    for doc in documents:
        body.append({
            "ranking": doc[0],
            "text": " ".join(doc[1].terms)
        })
    return body

@app.get("/query")
def read_item(value: str = ""):
    documents = vm.query(value)
    return prepare_output(documents)


