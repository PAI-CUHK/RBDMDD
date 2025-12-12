from __future__ import annotations
import torch
import torch.nn as nn
from slstm import sLSTM

class StatisticalPooling(nn.Module):
    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_len, hidden_dim = sequence.size()
        mask = (
            torch.arange(max_len, device=sequence.device)
            .expand(batch_size, max_len)
            < lengths.unsqueeze(1)
        ).unsqueeze(-1)
        mask = mask.float()
        masked = sequence * mask

        sum_out = masked.sum(dim=1)
        mean_pool = sum_out / lengths.unsqueeze(1).float().to(sequence.device)

        neg_inf = torch.full_like(masked, fill_value=-1e9)
        masked_for_max = torch.where(mask.bool(), masked, neg_inf)
        max_pool, _ = masked_for_max.max(dim=1)
        max_pool[max_pool == -1e9] = 0.0

        return torch.cat([mean_pool, max_pool], dim=1)


def flip_padded_sequences(inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

    batch_size, max_len, dim = inputs.shape
    flipped = torch.zeros_like(inputs)
    
    for i in range(batch_size):
        length = int(lengths[i])
        flipped[i, :length, :] = inputs[i, :length, :].flip(dims=[0])
    return flipped

class Audio(nn.Module):

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.net(x)


class TriStreamModel(nn.Module):
    def __init__(
        self,
        hog_dim: int = 256,
        au_dim: int = 17,
        input_dim: int = 128,
        visual_lstm_dim: int = 128,
        visual_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        visual_input_dim = hog_dim + au_dim
        
        self.visual_slstm_fwd = sLSTM(
            input_size=visual_input_dim,
            hidden_size=visual_lstm_dim,
            num_layers=visual_layers,
            dropout=0.5 if visual_layers > 1 else 0.0
        )
        
        self.visual_slstm_bwd = sLSTM(
            input_size=visual_input_dim,
            hidden_size=visual_lstm_dim,
            num_layers=visual_layers,
            dropout=0.5 if visual_layers > 1 else 0.0
        )
        
        
        self.audio_extractor = Audio(input_dim=input_dim, output_dim=64)
        self.audio_pooler = StatisticalPooling()

       
        fusion_dim = (visual_lstm_dim * 2) + (64 * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        hog_sequence: torch.Tensor,
        hog_lengths: torch.Tensor,
        au_sequence: torch.Tensor,
        au_lengths: torch.Tensor,
        audio_sequence: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.equal(hog_lengths, au_lengths):
            raise ValueError("HOG 和 AU 序列长度必须一致")


        visual_input = torch.cat([hog_sequence, au_sequence], dim=-1)
        batch_size, max_len, _ = visual_input.shape


        out_fwd, _ = self.visual_slstm_fwd(visual_input)
        

        visual_input_flipped = flip_padded_sequences(visual_input, hog_lengths)
        out_bwd, _ = self.visual_slstm_bwd(visual_input_flipped)

        row_indices = torch.arange(batch_size, device=visual_input.device)
        last_indices = (hog_lengths - 1).long().to(visual_input.device)
        
        fwd_vec = out_fwd[row_indices, last_indices, :]
        

        bwd_vec = out_bwd[row_indices, last_indices, :]
        
        visual_vec = torch.cat([fwd_vec, bwd_vec], dim=1)

        # --- Audio Stream  ---
        audio_feat = self.audio_extractor(audio_sequence)
        audio_vec = self.audio_pooler(audio_feat, audio_lengths)

        # --- Fusion ---
        fused = torch.cat([visual_vec, audio_vec], dim=1)
        return self.classifier(fused)