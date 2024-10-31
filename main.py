from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Define the allowed origins
origins = [
    "http://localhost",  # Allow this local origin
    "http://localhost:5500",  # Allow this local origin
    "http://127.0.0.1:5500",  # Replace with your frontend domain
    # Add more origins as needed
]

# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Origins allowed to make requests
    allow_credentials=True,  # Whether to allow credentials (e.g., cookies)
    allow_methods=["*"],  # HTTP methods allowed for CORS requests
    allow_headers=["*"],  # HTTP headers allowed for CORS requests
)

class InputStructure(BaseModel):
    location: float
    size: float
    total_sqft: float
    bath: float
    balcony: float



@app.post('/api/get_price')
async def get_price(item: InputStructure):
    with open('files/regression.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('files/std_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    x = scaler.transform(np.array([[item.location, item.size, item.total_sqft, item.bath, item.balcony]]))
    price = model.predict(x)[0]
    return {'price': price}

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html file at root URL
@app.get("/")
async def root():
    return FileResponse(Path("static/index.html"))
# uvicorn main:app --reload