import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm


def train_medbios_local(model, tokenizer, device):

    # =========================================================
    # 1. LOAD DATASETS PROPERLY
    # =========================================================
    train_dataset = get_medbios_dataset(tokenizer, split='train')
    val_dataset = get_medbios_dataset(tokenizer, split='validation')
    test_dataset = get_medbios_dataset(tokenizer, split='test')

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # =========================================================
    # 2. DATALOADERS
    # =========================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # =========================================================
    # 3. FREEZE MODEL
    # =========================================================
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification head
    model.lm_head = torch.nn.Linear(768, 5, bias=True).to(device)

    optimizer = AdamW(model.lm_head.parameters(), lr=7e-5)

    criterion = torch.nn.CrossEntropyLoss()

    scheduler = LinearLR(
        optimizer,
        start_factor=0.5,
        total_iters=5
    )

    # =========================================================
    # 4. TRAINING
    # =========================================================
    best_val_loss = float('inf')
    max_epochs = 10

    for epoch in range(max_epochs):

        # ---------------- TRAIN ----------------
        model.train()

        total_train_loss = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} [Train]"
        )

        for batch in pbar:

            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)

            pad_id = tokenizer.pad_token_id
            seq_lengths = (input_ids != pad_id).sum(dim=1) - 1
            logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
            
            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })

        avg_train_loss = total_train_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()

        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for batch in val_loader:

                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids)

                pad_id = tokenizer.pad_token_id
                seq_lengths = (input_ids != pad_id).sum(dim=1) - 1
                logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
                
                loss = criterion(logits, labels)

                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        scheduler.step()

        # ---------------- EARLY STOPPING ----------------
        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(
                model.state_dict(),
                'mamba_medbios_weights.pt'
            )

            print("Saved best model.")

        else:
            print("Validation loss stopped improving. Early stopping.")
            break

    # =========================================================
    # 5. FINAL TEST EVALUATION
    # =========================================================
    print("\nEvaluating on Test Set...")

    model.load_state_dict(
        torch.load('mamba_medbios_weights.pt')
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)

            pad_id = tokenizer.pad_token_id
            seq_lengths = (input_ids != pad_id).sum(dim=1) - 1
            logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
            
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total

    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


# =========================================================
# EXECUTION
# =========================================================
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

model.to(device)

train_medbios_local(model, tokenizer, device)