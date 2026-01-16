"""
Stage 3: Text decoder for generating text from EEG embeddings.

Architecture:
    EEG (frozen JEPA) -> Predicted Embedding -> BART Decoder -> Generated Text
"""

import torch
import torch.nn as nn
from typing import List, Optional
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput


class EEG2TextDecoder(nn.Module):
    """
    Stage 3: BART decoder for text generation from EEG embeddings.
    
    Uses frozen Stage 2 model to get EEG embeddings, then decodes to text.
    """
    
    def __init__(self,
                 jepa_model,  # Trained Stage 2 EEG2TextJEPA model
                 bart_model_name: str = 'facebook/bart-base',
                 freeze_jepa: bool = True,
                 max_length: int = 64):
        super().__init__()
        
        self.max_length = max_length
        
        # Freeze JEPA model (Stage 2)
        self.jepa_model = jepa_model
        if freeze_jepa:
            for param in self.jepa_model.parameters():
                param.requires_grad = False
        
        # BART decoder
        print(f"Loading BART: {bart_model_name}")
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)
        
        # Project EEG embedding to BART encoder hidden size
        self.embed_proj = nn.Sequential(
            nn.Linear(jepa_model.text_embed_dim, self.bart.config.d_model),
            nn.LayerNorm(self.bart.config.d_model),
            nn.GELU(),
            nn.Linear(self.bart.config.d_model, self.bart.config.d_model),
            nn.LayerNorm(self.bart.config.d_model)
        )
    
    def forward(self, 
                eeg: torch.Tensor, 
                labels: Optional[torch.Tensor] = None,
                texts: Optional[List[str]] = None):
        """
        Forward pass for training.
        
        Args:
            eeg: EEG tensor (B, 105, 8)
            labels: Token IDs for teacher forcing (B, seq_len)
            texts: Optional text strings (will be tokenized to labels)
            
        Returns:
            outputs: BART model outputs with loss
        """
        # Get EEG embedding from frozen JEPA
        with torch.no_grad():
            eeg_embed, _, _ = self.jepa_model(eeg)  # (B, text_embed_dim)
        
        # Project to BART dimension
        encoder_hidden = self.embed_proj(eeg_embed).unsqueeze(1)  # (B, 1, d_model)
        
        # Tokenize texts if provided
        if labels is None and texts is not None:
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            labels = tokenized['input_ids'].to(eeg.device)
        
        # Create dummy encoder attention mask
        encoder_attention_mask = torch.ones(
            encoder_hidden.size(0), 1,
            device=encoder_hidden.device,
            dtype=torch.long
        )
        
        # Wrap encoder outputs in BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        
        # BART forward with encoder outputs
        outputs = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(self,
                 eeg: torch.Tensor,
                 max_length: int = 50,
                 num_beams: int = 5,
                 early_stopping: bool = True,
                 **generate_kwargs) -> List[str]:
        """
        Generate text from EEG.
        
        Args:
            eeg: EEG tensor (B, 105, 8)
            max_length: Maximum generation length
            num_beams: Beam search width
            early_stopping: Stop when EOS token is generated
            
        Returns:
            Generated text strings
        """
        self.eval()
        
        # Get EEG embedding
        eeg_embed, _, _ = self.jepa_model(eeg)  # (B, text_embed_dim)
        
        # Project to BART dimension
        encoder_hidden = self.embed_proj(eeg_embed).unsqueeze(1)  # (B, 1, d_model)
        
        # Create encoder attention mask
        encoder_attention_mask = torch.ones(
            encoder_hidden.size(0), 1,
            device=encoder_hidden.device,
            dtype=torch.long
        )
        
        # Wrap encoder outputs in BaseModelOutput (required by transformers)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        
        # Generate
        generated_ids = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            **generate_kwargs
        )
        
        # Decode to text
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return texts

