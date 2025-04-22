import os
from transformers import (AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_ligand_data(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def prepare_mock_dataset_chunk(ligands_chunk, sep_token, max_length):
    samples = []
    current_sample = ""

    for ligand in tqdm(ligands_chunk):
        ligand_with_sep = ligand + sep_token
        if len(current_sample + ligand_with_sep) > max_length:
            samples.append(current_sample.strip())
            current_sample = ligand_with_sep
        else:
            current_sample += ligand_with_sep

    if current_sample:
        samples.append(current_sample.strip())

    return samples

def prepare_mock_dataset_parallel(ligands, sep_token="[SEP]", max_length=512, num_processes=None):
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    chunk_size = (len(ligands) + num_processes - 1) // num_processes
    chunks = [ligands[i:i + chunk_size] for i in range(0, len(ligands), chunk_size)]

    mock_sequences = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(prepare_mock_dataset_chunk, chunk, sep_token, max_length) for chunk in chunks]
        for future in tqdm(futures, desc="Processing Chunks"):
            mock_sequences.extend(future.result())

    return mock_sequences

class LigandDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}

def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

def train_model(model, tokenizer, train_dataset, eval_dataset, model_name, output_dir, max_length, batch_size, epochs, lr):
    data_collator = create_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        optim="adamw_torch_fused",
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        prediction_loss_only=False,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        ddp_find_unused_parameters=False,
        torch_compile=False,
        lr_scheduler_type="cosine",
        learning_rate=lr,
        report_to='all',
        bf16=True,
        tf32=True,
        seed=42,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    FILE_PATH = "77M-Ligands.txt"
    with open(FILE_PATH, "r") as file:
        smiles = file.read().split("\n")
        
    sequences = prepare_mock_dataset_chunk(smiles, sep_token="[SEP]", max_length=512)

    from transformers import RobertaConfig, AutoConfig, AutoTokenizer
    from DTI import FlashRobertaForMaskedLM
    model_id = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    config.num_attention_heads = 24
    config.hidden_size = config.hidden_size
    config.max_position_embeddings = 515
    config.intermediate_size = config.hidden_size * 2
    model = FlashRobertaForMaskedLM(config)
    OUTPUT_DIR = "./weights/OurNewMoleculeModel-v1"
    train, val = train_test_split(sequences, test_size=0.001, random_state=42)
    trainset = LigandDataset(train, tokenizer, 512)
    valset = LigandDataset(val, tokenizer, 512)

    MAX_LENGTH = 512
    BATCH_SIZE = 512
    EPOCHS = 50
    LEARNING_RATE = 5e-5
    train_model(model, tokenizer, trainset, valset, model_id, OUTPUT_DIR, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE)