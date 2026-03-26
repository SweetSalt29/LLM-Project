from pydantic import BaseModel

class UserRegister(BaseModel):
    name: str
    password: str

class UserOut(BaseModel):
    id: int
    name: str

class Token(BaseModel):
    access_token: str
    token_type: str

class QueryRequest(BaseModel):
    query: str