import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from mamba_lrp.dataset.general_dataset import get_medbios_dataset
from mamba_ssm.models.modeling_mamba import MambaLMHeadModel

def create_medbios_mamba_weights(model, tokenizer):
    # 1. Setup Model for 28 classes (Med-BIOS)
    model.lm_head = torch.nn.Linear(768, 28, bias=True).to(device)
    train_dataset = get_medbios_dataset(tokenizer, split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # 2. Simple Training Loop
    model.train()
    for epoch in range(3):  # Adjust epochs as needed
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Mamba returns CausalLMOutput; we need the last token's logits[cite: 2]
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] 
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")

    # 3. Save your new weights
    torch.save(model.state_dict(), 'mamba_medbios_weights.pt')