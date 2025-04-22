import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_data(test_size, seed):
    data = load_dataset("amirhallaji/davis")["train"].to_pandas()
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=seed)
    
    train_data = (
        train_df["Molecule Sequence"].tolist(),
        train_df["Protein Sequence"].tolist(),
        train_df["Binding Affinity"].tolist(),
    )
    
    test_data = (
        test_df["Molecule Sequence"].tolist(),
        test_df["Protein Sequence"].tolist(),
        test_df["Binding Affinity"].tolist(),
    )
    
    return train_data, test_data


class DrugTargetDataset(Dataset):
    def __init__(self, molecules, proteins, labels):
        self.molecules = molecules
        self.proteins = proteins
        self.labels = labels
        
    def __len__(self):
        return len(self.molecules)
        
    def __getitem__(self, idx):
        return self.molecules[idx], self.proteins[idx], self.labels[idx]


def collate_fn(batch, molecule_tokenizer, protein_tokenizer, max_global_molecule_length=128, max_global_protein_length=1024):
    molecules, proteins, labels = zip(*batch)
    
    batch_max_molecule_length = min(max(len(m) for m in molecules), max_global_molecule_length)
    batch_max_protein_length = min(max(len(p) for p in proteins), max_global_protein_length)
    
    molecule_tokens = molecule_tokenizer(
        list(molecules),
        padding=True,
        truncation=True,
        max_length=batch_max_molecule_length,
        return_tensors="pt"
    )
    
    protein_tokens = protein_tokenizer(
        list(proteins),
        padding=True,
        truncation=True,
        max_length=batch_max_protein_length,
        return_tensors="pt"
    )
    
    labels = torch.tensor(labels, dtype=torch.float)
    
    return {
        "molecule_input_ids": molecule_tokens["input_ids"],
        "molecule_attention_mask": molecule_tokens["attention_mask"],
        "protein_input_ids": protein_tokens["input_ids"],
        "protein_attention_mask": protein_tokens["attention_mask"],
        "label": labels
    }


def get_data_loaders(config):
    (train_mol, train_prot, train_labels), (test_mol, test_prot, test_labels) = load_data(
        config.test_size, config.seed
    )
    
    train_dataset = DrugTargetDataset(train_mol, train_prot, train_labels)
    test_dataset = DrugTargetDataset(test_mol, test_prot, test_labels)
    
    protein_tokenizer = AutoTokenizer.from_pretrained(config.protein_model_name)
    molecule_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model_name)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, molecule_tokenizer, protein_tokenizer)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, molecule_tokenizer, protein_tokenizer)
    )
    
    return train_loader, test_loader