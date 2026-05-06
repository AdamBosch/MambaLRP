import torch
from torch.utils.data import DataLoader
from transformers import AdamW

# 1. Define the training logic directly in the notebook to avoid import errors
def train_medbios_local(model, tokenizer, device):
    # Use the function we defined earlier for the medical dataset
    train_dataset = get_medbios_dataset(tokenizer, split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] 
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")

    torch.save(model.state_dict(), 'mamba_medbios_weights.pt')
    print("Weights saved to mamba_medbios_weights.pt")

# 2. Run it
# 1. Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Update the head for 28 classes
model.lm_head = torch.nn.Linear(768, 28, bias=True)

# 3. CRITICAL: Move the ENTIRE model to the GPU
model.to(device)

# 4. Now run the training
train_medbios_local(model, tokenizer, device)