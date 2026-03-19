import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Model Architecture ---
class PetRockGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(PetRockGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # GRU is used to fight the Vanishing Gradient problem.
        # It uses gates to decide which information to keep or forget.
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # Training Head: We need this to calculate loss during training.
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.gru(x, h)
        logits = self.fc(out)
        return logits, h

# --- 2. Setup Hyperparameters ---
VOCAB_SIZE = 100   # Unique "thoughts" the rock can have
EMBED_DIM = 32     # Size of word vectors
HIDDEN_DIM = 64    # Size of the rock's internal memory
SEQ_LEN = 15       # Length of sentences the rock listens to
EPOCHS = 100       # How many times the rock studies
BATCH_SIZE = 16

# --- 3. Generate Dummy Training Data ---
# We create random sequences and shift them by one to teach "Next Token Prediction"
data_in = torch.randint(0, VOCAB_SIZE, (100, SEQ_LEN))
data_target = torch.roll(data_in, -1, dims=1)

# --- 4. Training Initialization ---
model = PetRockGRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_history = []

print("Starting Training (Pet Rock is contemplating the universe...)\n")

# --- 5. The Training Loop ---
plt.ion() # Turn on interactive mode for live plotting
fig, ax = plt.subplots()
line, = ax.plot([], [], color='brown', label='Learning Progress')
ax.set_title("Pet Rock Training: Overcoming Vanishing Gradients")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (Confusion Level)")

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # Forward Pass
    logits, _ = model(data_in)
    
    # Calculate Loss (Cross-Entropy)
    loss = criterion(logits.view(-1, VOCAB_SIZE), data_target.view(-1))
    
    # Backward Pass (BPTT - Backpropagation Through Time)
    loss.backward()
    
    # Update Weights
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.4f}")
    
    # Update Graph
    line.set_data(range(len(loss_history)), loss_history)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

plt.ioff()
print("\nTraining Complete! The rock is now enlightened.")
plt.show()

# --- 6. The Stoic Inference Mode ---
def pet_rock_respond(user_text):
    # This represents the model 'understanding' but choosing the icon
    model.eval()
    return "🪨"

print(f"User: 'Hello Rock!' -> Rock: {pet_rock_respond('Hello')}")

torch.save(model.state_dict(), "pet_rock_brain.pth")