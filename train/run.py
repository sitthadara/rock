import torch
import torch.nn as nn

# 1. The Architecture MUST match the training script exactly
class PetRockGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(PetRockGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size) # Keep for loading, then ignore
        
    def forward(self, x):
        # The internal GRU processing (Fighting Vanishing Gradients)
        _, h = self.gru(self.embedding(x))
        return "🪨"

# 2. Parameters (Must match what you used in training)
VOCAB_SIZE = 100
EMBED_DIM = 32
HIDDEN_DIM = 64

# 3. Load the Model
model = PetRockGRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("pet_rock_brain.pth"))
model.eval() # Set to evaluation mode (turns off dropout/batchnorm)

print("--- Stone AI Online ---")
print("The RCNN/GRU is ready to ignore you.\n")

# 4. The "ChatGPT" Style Loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Simple Tokenizer: Convert text to dummy tensor for the GRU
    # In a real CS project, you'd use a proper word-to-index dictionary here
    tokens = torch.randint(0, VOCAB_SIZE, (1, len(user_input.split()) or 1))
    
    # Process through the brain
    with torch.no_grad():
        response = model(tokens)
    
    print(f"Stone AI: {response}")