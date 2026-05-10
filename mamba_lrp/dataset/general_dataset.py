from torch.utils.data import Dataset
from datasets import load_dataset


class GeneralDataset(Dataset):
    def __init__(
            self,
            inputs,
            targets,
            tokenizer,
            max_length=None,
            truncation=False
    ):
        self.inputs = ["<|startoftext|>" + inp + "<|endoftext|>" for inp in inputs]
        self.targets = targets
        self.tokenizer = tokenizer
        self.num_classes = len(set(self.targets))

        # Tokenize the input
        if max_length:
            self.tokenized_inputs = tokenizer(
                self.inputs,
                padding=True,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=max_length,
                truncation=truncation
            )
        else:
            self.tokenized_inputs = tokenizer(
                self.inputs,
                padding=True,
                return_tensors="pt",
                add_special_tokens=True
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        features = {'input_ids': self.tokenized_inputs.input_ids[idx],
                    'attention_mask': self.tokenized_inputs.attention_mask[idx],
                    'label': self.targets[idx]
                    }
        return features


def get_sst_dataset(
        tokenizer,
        max_length,
        truncation,
        split='val'
):
    dataset = load_dataset("glue", 'sst2')

    if split == 'train':
        dataset = GeneralDataset(
            dataset["train"]["sentence"],
            dataset["train"]["label"],
            tokenizer,
            max_length,
            truncation
        )
    elif split == 'val':
        dataset = GeneralDataset(
            dataset["validation"]["sentence"],
            dataset["validation"]["label"],
            tokenizer,
            max_length,
            truncation
        )

    return dataset

def get_medbios_dataset(
        tokenizer,
        max_length=512,
        truncation=True,
        split='test'
):
    dataset = load_dataset("coastalcph/medical-bios", "standard")

    if split == 'train':
        data = dataset["train"]
    elif split == 'validation':
        data = dataset["validation"]
    else:
        data = dataset["test"]
    return GeneralDataset(
        inputs=data["text"],
        targets=data["label"],
        tokenizer=tokenizer,
        max_length=max_length,
        truncation=truncation
    )

def get_bias_in_bios_dataset(
        tokenizer,
        max_length=512,
        truncation=True,
        split='test',
        max_samples = None
):
    dataset = load_dataset("LabHC/bias_in_bios", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))
    return GeneralDataset(
        inputs=dataset["hard_text"],
        targets=dataset["profession"],
        tokenizer=tokenizer,
        max_length=max_length,
        truncation=truncation
    )
    

def get_emotion_dataset(
    tokenizer,
    max_length=512,
    truncation=True,
    split='test'
):
    dataset = load_dataset("philschmid/emotion", split=split)
    return GeneralDataset(
        inputs=dataset["text"],
        targets=dataset["label"],
        tokenizer=tokenizer,
        max_length=max_length,
        truncation=truncation
    )

def get_snli_dataset(
    tokenizer,
    max_length=512,
    truncation=True,
    split="test"
):
    data = load_dataset("stanfordnlp/snli", split=split)

    valid_data = data.filter(lambda x: x["label"] != -1)

    inputs = [
        f"premise: {premise} hypothesis: {hypothesis}"
        for premise, hypothesis in zip(
            valid_data["premise"],
            valid_data["hypothesis"]
        )
    ]

    return GeneralDataset(
        inputs=inputs,
        targets=valid_data["label"],
        tokenizer=tokenizer,
        max_length=max_length,
        truncation=truncation
    )

class ImageNetDataset(Dataset):
    def __init__(self, split="validation", transform=None, max_samples=None):
        self.data = load_dataset("imagenet-1k", split=split)
        if max_samples is not None:
            self.data = self.data.select(range(max_samples))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.data[i]["image"].convert("RGB")   # PIL → RGB
        label = self.data[i]["label"]
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": label}

def get_imagenet_dataset(transform, split="validation", max_samples=None, n_classes=1000):
    # n_classes kept as parameter for API consistency — Vim-S always has 1000
    assert n_classes == 1000, "Vim-S is pretrained on 1000 ImageNet classes"
    return ImageNetDataset(split=split, transform=transform, max_samples=max_samples)
