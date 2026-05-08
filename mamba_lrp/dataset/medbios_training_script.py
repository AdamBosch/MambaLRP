import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from transformers import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

def train_medbios_local(model, tokenizer, device):
    # 1. LOAD AND COMBINE DATASETS
    # Load multiple splits to reach the 10,000 sample requirement
    train_pool = get_medbios_dataset(tokenizer, split='train') # Likely 8,000 samples
    eval_pool = get_medbios_dataset(tokenizer, split='dev')
    test_pool = get_medbios_dataset(tokenizer, split='test')   # Typically 2,000 samples
    
    # Combine them into one large dataset of ~10,000
    full_dataset = ConcatDataset([train_pool, test_pool, eval_pool])
    total_size = len(full_dataset)
    print(f"Total dataset size: {total_size}")

    # Define requested sizes
    train_size, val_size, test_size = 7200, 900, 900
    
    # Safety check: Adjust if the dataset is smaller than 10k
    if total_size < (train_size + val_size + test_size):
        print(f"Warning: Dataset size ({total_size}) is smaller than requested 10k.")
        # Optional: Proportional split logic here if needed
    
    # 2. CREATE SUBSETS
    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(full_dataset, range(train_size + val_size, train_size + val_size + test_size))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. SETUP MODEL & OPTIMIZER
    for param in model.parameters():
        param.requires_grad = False
    
    model.lm_head = torch.nn.Linear(768, 5, bias=True).to(device)
    optimizer = AdamW(model.lm_head.parameters(), lr=7e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Linear scheduler with initial factor of 0.5
    # total_iters should ideally match the number of steps or epochs
    scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=5)

    best_val_loss = float('inf')
    max_epochs = 10

    for epoch in range(max_epochs):
        # --- TRAINING ---
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] 
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]
                total_val_loss += criterion(logits, labels).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step()

        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'mamba_medbios_weights.pt')
        else:
            print("Validation loss stopped improving. Stopping early.")
            break

    # --- FINAL TEST ---
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load('mamba_medbios_best.pt'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    print(f"Final Test Accuracy: {(correct/total)*100:.2f}%")

# Execution block
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_medbios_local(model, tokenizer, device)