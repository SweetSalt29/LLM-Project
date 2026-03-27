from fastapi import FastAPI
from backend.api import auth, upload, query

app = FastAPI()

app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(query.router)


@app.get("/")
def home():
    return {"message": "Welcome to Document and Data Analyzer!"}