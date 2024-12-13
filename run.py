from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)