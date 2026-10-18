# import torch
# import torch.nn as nn

# # 1. The Architecture MUST match the training script exactly
# class PetRockGRU(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim):
#         super(PetRockGRU, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, vocab_size) # Keep for loading, then ignore
        
#     def forward(self, x):
#         # The internal GRU processing (Fighting Vanishing Gradients)
#         _, h = self.gru(self.embedding(x))
#         return "🪨"

# # 2. Parameters (Must match what you used in training)
# VOCAB_SIZE = 100
# EMBED_DIM = 32
# HIDDEN_DIM = 64

# # 3. Load the Model
# model = PetRockGRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
# model.load_state_dict(torch.load("pet_rock_brain.pth"))
# model.eval() # Set to evaluation mode (turns off dropout/batchnorm)

# print("--- Stone AI Online ---")
# print("The RCNN/GRU is ready to ignore you.\n")

# # 4. The "ChatGPT" Style Loop
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         break
    
#     # Simple Tokenizer: Convert text to dummy tensor for the GRU
#     # In a real CS project, you'd use a proper word-to-index dictionary here
#     tokens = torch.randint(0, VOCAB_SIZE, (1, len(user_input.split()) or 1))
    
#     # Process through the brain
#     with torch.no_grad():
#         response = model(tokens)
    
#     print(f"Stone AI: {response}")

import torch
import torch.nn as nn

# =========================
# 1. Vocabulary (MUST match training)
# =========================
vocab = {
    "hello": 0,
    "how": 1,
    "are": 2,
    "you": 3,
    "what": 4,
    "is": 5,
    "life": 6,
    "tell": 7,
    "me": 8,
    "something": 9,
    "fine": 10,
    "ok": 11,
    "🪨": 12,
    "<unk>": 13
}

idx_to_token = {v: k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)

# =========================
# Model
# =========================
class PetRockGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        out = self.fc(h.squeeze(0))
        return out

# =========================
# 3. Parameters (same!)
# =========================
EMBED_DIM = 32
HIDDEN_DIM = 64

# =========================
# 4. Load Model
# =========================
model = PetRockGRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("pet_rock_brain.pth"))
model.eval()

# =========================
# 5. Tokenizer
# =========================
def encode(text):
    return [vocab.get(word, vocab["<unk>"]) for word in text.lower().split()]

def decode(idx):
    return idx_to_token.get(idx, "<unk>")

# =========================
# 6. Predict Function
# =========================
def predict(text):
    tokens = torch.tensor([encode(text)])

    with torch.no_grad():
        output = model(tokens)
        probs = torch.softmax(output, dim=1)

        # Option 1: deterministic
        pred = torch.argmax(probs, dim=1).item()

        # Option 2 (more fun): sampling
        # pred = torch.multinomial(probs, 1).item()

    return decode(pred)

# =========================
# 7. Chat Loop
# =========================
print("🪨 Pet Rock AI is online. It may or may not care.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    response = predict(user_input)
    print(f"Stone AI: {response}")