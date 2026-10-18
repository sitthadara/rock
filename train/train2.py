import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =========================
# 1. Vocabulary
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
# 2. Model
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
# 3. Hyperparameters
# =========================
EMBED_DIM = 32
HIDDEN_DIM = 64
EPOCHS = 1000
LR = 0.001

model = PetRockGRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# 4. Training Data
# =========================
training_data = [
    ("hello", "🪨"),
    ("how are you", "🪨"),
    ("what is life", "🪨"),
    ("tell me something", "🪨"),
    ("how are you", "🪨"),
    ("tell me something", "🪨"),
]

def encode(text):
    return [vocab.get(word, vocab["<unk>"]) for word in text.split()]

# =========================
# 5. Training Loop + Logging
# =========================
loss_history = []

for epoch in range(EPOCHS):
    total_loss = 0

    for text, target in training_data:
        tokens = torch.tensor([encode(text)])
        target_idx = torch.tensor([vocab[target]])

        output = model(tokens)
        loss = criterion(output, target_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(training_data)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# =========================
# 6. Save Model
# =========================
torch.save(model.state_dict(), "pet_rock_brain.pth")

print("✅ Training complete. Model saved.")

# =========================
# 7. Plot Graph
# =========================
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()