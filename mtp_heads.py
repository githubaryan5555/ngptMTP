"""
Multi-Token Prediction (MTP) Heads Module

This module provides clean, modular MTP prediction heads that predict
multiple future tokens (k=1,2,3,...) in parallel during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """
    A single prediction head for predicting k steps ahead.
    
    Predicts the k-th token from the transformer hidden states.
    Uses a simple linear projection to vocab_size.
    """
    
    def __init__(self, n_embd: int, vocab_size: int, k: int = 1):
        """
        Args:
            n_embd: Embedding dimension from transformer
            vocab_size: Size of vocabulary
            k: How many steps ahead this head predicts (1-indexed)
        """
        super().__init__()
        self.k = k
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, n_embd)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return self.head(hidden_states)


class MTPHeads(nn.Module):
    """
    Container for multiple MTP heads predicting k=1,2,3,...,multitok_pred steps ahead.
    
    All heads share the same transformer backbone and only the output layers differ.
    This is more efficient than training separate models.
    """
    
    def __init__(self, n_embd: int, vocab_size: int, multitok_pred: int):
        """
        Args:
            n_embd: Embedding dimension from transformer
            vocab_size: Size of vocabulary
            multitok_pred: Number of future tokens to predict (K)
        """
        super().__init__()
        self.multitok_pred = multitok_pred
        self.vocab_size = vocab_size
        
        # Create K separate heads: one for each prediction offset
        self.heads = nn.ModuleList([
            MTPHead(n_embd, vocab_size, k=k)
            for k in range(1, multitok_pred + 1)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> list:
        """
        Args:
            hidden_states: (batch, seq_len, n_embd)
        
        Returns:
            List of logits, one per prediction head:
            [(batch, seq_len, vocab_size), ...]
        """
        return [head(hidden_states) for head in self.heads]


class MTPLoss(nn.Module):
    """
    Computes Multi-Token Prediction loss.
    
    For each prediction head k (predicting k steps ahead):
        - Use logits from positions 0..T-k to predict targets at positions k..T
        - Compute cross-entropy loss
        - Average or weight the losses across all k
    
    This encourages the model to build better intermediate representations
    that help predict further into the future.
    """
    
    def __init__(self, multitok_pred: int, reduction: str = 'mean', 
                 per_k_weights: list = None):
        """
        Args:
            multitok_pred: Number of future tokens to predict
            reduction: 'mean' or 'sum'
            per_k_weights: Optional list of weights for each k. If None, equal weight.
                          If provided, should have length multitok_pred.
        """
        super().__init__()
        self.multitok_pred = multitok_pred
        self.reduction = reduction
        
        if per_k_weights is None:
            # Equal weight for all k by default
            self.per_k_weights = [1.0 / multitok_pred for _ in range(multitok_pred)]
        else:
            assert len(per_k_weights) == multitok_pred
            # Normalize weights to sum to 1
            total = sum(per_k_weights)
            self.per_k_weights = [w / total for w in per_k_weights]
        
        self.per_k_weights = torch.tensor(self.per_k_weights, dtype=torch.float32)
    
    def forward(self, logits_list: list, targets: torch.Tensor) -> tuple:
        """
        Compute MTP loss.
        
        Args:
            logits_list: List of logits from MTP heads, each (batch, seq_len, vocab_size)
            targets: (batch, seq_len) - ground truth token IDs
        
        Returns:
            loss: scalar tensor (weighted average across all k)
            loss_per_k: list of losses for each k (for logging)
        """
        batch_size, seq_len, vocab_size = logits_list[0].shape
        device = logits_list[0].device
        
        # Move weights to device
        weights = self.per_k_weights.to(device)
        
        loss_per_k = []
        total_loss = torch.tensor(0.0, device=device, dtype=logits_list[0].dtype)
        
        for k, logits in enumerate(logits_list, start=1):
            # logits: (batch, seq_len, vocab_size)
            # targets: (batch, seq_len)
            
            # For k-th prediction: use positions 0..seq_len-k to predict positions k..seq_len
            if seq_len < k:
                # Not enough sequence length for this k
                loss_per_k.append(torch.tensor(0.0, device=device))
                continue
            
            # Slice: predict at positions that have valid targets k steps ahead
            pred_positions = logits[:, :-k, :]  # (batch, seq_len-k, vocab_size)
            target_positions = targets[:, k:]    # (batch, seq_len-k)
            
            # Flatten for cross-entropy
            pred_flat = pred_positions.reshape(-1, vocab_size)
            target_flat = target_positions.reshape(-1)
            
            # Compute cross-entropy (ignore_index=-1 for padding if needed)
            loss_k = F.cross_entropy(pred_flat, target_flat, ignore_index=-1, reduction='mean')
            loss_per_k.append(loss_k)
            
            # Accumulate weighted loss
            total_loss = total_loss + weights[k - 1] * loss_k
        
        # Return total loss and per-k losses for logging
        return total_loss, loss_per_k


def create_mtp_loss(multitok_pred: int, loss_weighting: str = 'uniform') -> MTPLoss:
    """
    Factory function to create MTP loss with different weighting schemes.
    
    Args:
        multitok_pred: Number of future tokens to predict
        loss_weighting: 'uniform' (equal weight) or 'linear' (linearly decay with k)
    
    Returns:
        MTPLoss instance
    """
    if loss_weighting == 'uniform':
        weights = None
    elif loss_weighting == 'linear':
        # Linear decay: more weight on close predictions (k=1), less on far (k=multitok_pred)
        weights = [float(multitok_pred - k + 1) for k in range(1, multitok_pred + 1)]
    else:
        raise ValueError(f"Unknown weighting scheme: {loss_weighting}")
    
    return MTPLoss(multitok_pred, per_k_weights=weights)
