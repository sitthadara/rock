from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import time
import random

app = FastAPI()

# --- 1. Model Architecture (Must match your training) ---
class PetRockGRU(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, hidden_dim=64):
        super(PetRockGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        return "🪨"

# --- 2. Load the Stone ---
model = PetRockGRU()
try:
    model.load_state_dict(torch.load("pet_rock_brain.pth"))
    model.eval()
except:
    print("Warning: .pth not found. Using untrained stone.")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    time.sleep(random.random()*2) 
    tokens = torch.randint(0, 100, (1, max(1, len(request.message.split()))))
    with torch.no_grad():
        response = model(tokens)
        
    return {"response": response}