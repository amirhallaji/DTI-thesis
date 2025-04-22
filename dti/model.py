import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from modern_bert import FlashRobertaForMaskedLM

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class ResidualInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], dropout=0.0):
        super(ResidualInceptionBlock, self).__init__()

        self.out_channels = out_channels
        num_branches = len(kernel_sizes)
        branch_out_channels = out_channels // num_branches

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels, branch_out_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(branch_out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])

        self.residual_adjust = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        residual = self.residual_adjust(x)
        output = self.relu(concatenated + residual)
        return output


class AffinityPredictor(nn.Module):
    def __init__(self, 
                 protein_model_name="facebook/esm2_t6_8M_UR50D", 
                 molecule_model_name="amirhallaji/apexchembert",
                 hidden_sizes=[1024, 768, 512, 256, 1], 
                 inception_out_channels=256,
                 dropout=0.00):
        super(AffinityPredictor, self).__init__()

        self.protein_model = AutoModel.from_pretrained(protein_model_name)
        self.molecule_model = FlashRobertaForMaskedLM.from_pretrained(molecule_model_name)

        self.protein_model.config.gradient_checkpointing = True
        self.protein_model.gradient_checkpointing_enable()

        model_id = molecule_model_name
        config = AutoConfig.from_pretrained(model_id)
        config.num_attention_heads = 24
        config.max_position_embeddings = 515
        config.intermediate_size = config.hidden_size * 3
        config.num_hidden_layers = 6

        self.molecule_model = FlashRobertaForMaskedLM(config)
        self.molecule_model.lm_head = torch.nn.Identity()
        self.molecule_model.config.gradient_checkpointing = True
        self.molecule_model.gradient_checkpointing_enable()
        
        prot_embedding_dim = self.protein_model.config.hidden_size
        mol_embedding_dim = self.molecule_model.config.hidden_size
        combined_dim = prot_embedding_dim + mol_embedding_dim

        self.inc1 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)
        self.inc2 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)

        layers = []
        input_dim = combined_dim
        for output_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, output_dim))
            if output_dim != 1:
                layers.append(Mish())
            input_dim = output_dim
        self.regressor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        protein_input = {
            "input_ids": batch["protein_input_ids"],
            "attention_mask": batch["protein_attention_mask"]
        }
        molecule_input = {
            "input_ids": batch["molecule_input_ids"],
            "attention_mask": batch["molecule_attention_mask"]
        }
        protein_embedding = self.protein_model(**protein_input).last_hidden_state.mean(dim=1)  # (batch_size, hidden_dim)
        molecule_embedding = self.molecule_model(**molecule_input).logits.mean(dim=1)  # (batch_size, hidden_dim)
        combined_features = torch.cat((protein_embedding, molecule_embedding), dim=1).unsqueeze(2)  # (batch_size, combined_dim, 1)
        combined_features = self.inc1(combined_features)  # (batch_size, combined_dim)
        combined_features = self.inc2(combined_features)
        combined_features = combined_features.squeeze(2)
        output = self.regressor(self.dropout(combined_features))  # (batch_size, 1)
        return output


class DrugTargetInteractionLoss(nn.Module):
    def __init__(self, alpha=0.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction="sum")
        
    def forward(self, pred, interaction_score):
        cosine_loss = 1 - F.cosine_similarity(pred, interaction_score, dim=-1)
        mse_loss = self.mse_loss(pred, interaction_score)
        total_loss = self.alpha * cosine_loss + (1 - self.alpha) * mse_loss
        if self.reduction == 'mean':
            return total_loss.mean()
        if self.reduction == 'sum':
            return total_loss.sum()
        return total_loss