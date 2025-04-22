from dataclasses import dataclass, field


@dataclass
class Config:
    protein_model_name: str = "facebook/esm2_t6_8M_UR50D"
    molecule_model_name: str = "amirhallaji/apexchembert"
    tokenizer_model_name: str = "DeepChem/ChemBERTa-77M-MLM"
    hidden_sizes: tuple = (2048, 1024, 768, 512, 256, 1)
    dropout: float = 0.01
    
    batch_size: int = 64
    num_workers: int = 16
    test_size: float = 0.2
    seed: int = 0
    
    lr: float = 5e-5
    weight_decay: float = 1e-5
    num_warmup_steps: int = 10
    num_training_steps: int = 100
    accumulation_steps: int = 32
    epochs: int = 500
    patience: int = 100
    
    train_log_file: str = "training_log-Kiba.txt"
    model_save_path: str = "Davis-test.pth"
    final_model_path: str = "final_model.pth"
    
    kernel_sizes: list = field(default_factory=lambda: [1, 3])
    normalization: str = "batch"
    activation: str = "gelu"
    use_se: bool = True
    se_reduction_ratio: int = 8
    use_stochastic_depth: bool = True
    survival_prob: float = 0.9