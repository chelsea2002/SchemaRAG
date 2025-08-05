import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from FlagEmbedding import FlagModel
from typing import List, Dict, Tuple, Optional
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

class SafeMultiheadAttention(nn.Module):
    """Safe version of MultiheadAttention that handles edge cases"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.embed_dim = embed_dim
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """Safe attention computation with NaN handling"""
        batch_size = query.size(0)
        
        # Check if all sequences are masked
        if key_padding_mask is not None:
            # If all keys are masked for any sample, handle separately
            all_masked = key_padding_mask.all(dim=-1)  # [batch_size]
            
            if all_masked.any():
                # Create output tensor
                attn_output = torch.zeros_like(query)
                attn_weights = None
                
                # Process samples with valid keys
                valid_samples = ~all_masked
                if valid_samples.any():
                    try:
                        valid_query = query[valid_samples]
                        valid_key = key[valid_samples]
                        valid_value = value[valid_samples]
                        valid_mask = key_padding_mask[valid_samples]
                        
                        valid_output, valid_weights = self.attention(
                            valid_query, valid_key, valid_value,
                            key_padding_mask=valid_mask,
                            attn_mask=attn_mask
                        )
                        
                        attn_output[valid_samples] = valid_output
                        attn_weights = valid_weights
                        
                    except Exception as e:
                        print(f"Error in attention computation: {e}")
                        # Fallback: return query as-is
                        attn_output = query.clone()
                
                return attn_output, attn_weights
        
        # Normal case: use standard attention
        try:
            return self.attention(query, key, value, 
                                key_padding_mask=key_padding_mask,
                                attn_mask=attn_mask)
        except Exception as e:
            print(f"Error in attention computation: {e}")
            # Fallback: return query as-is
            return query.clone(), None


class SchemaAwareModel(nn.Module):
    """Schema-Aware Representation Learning Model with robust NaN handling"""
    def __init__(self, 
                 embed_dim: int = 1024, 
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super(SchemaAwareModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Use safe attention modules
        self.table_column_attention = SafeMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.question_table_attention = SafeMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.table_proj = nn.Linear(embed_dim, embed_dim)
        self.column_proj = nn.Linear(embed_dim, embed_dim)
        self.question_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _safe_layer_norm(self, x, layer_norm, residual=None):
        """Safe layer normalization with NaN checking"""
        try:
            if residual is not None:
                x = x + residual
            
            # Check for NaN or Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Warning: NaN/Inf detected before layer norm, using residual")
                return residual if residual is not None else torch.zeros_like(x)
            
            output = layer_norm(x)
            
            # Check output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("Warning: NaN/Inf detected after layer norm, using residual")
                return residual if residual is not None else torch.zeros_like(x)
            
            return output
            
        except Exception as e:
            print(f"Error in layer norm: {e}")
            return residual if residual is not None else torch.zeros_like(x)
    
    def forward(self, 
                question_embed: torch.Tensor,
                table_embeds: torch.Tensor,
                column_embeds: torch.Tensor,
                table_masks: torch.Tensor,
                column_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            question_embed: [batch_size, embed_dim]
            table_embeds: [batch_size, max_tables, embed_dim]
            column_embeds: [batch_size, max_tables, max_columns, embed_dim]
            table_masks: [batch_size, max_tables]
            column_masks: [batch_size, max_tables, max_columns]
        Returns:
            schema_aware_embed: [batch_size, embed_dim]
        """
        batch_size, max_tables, embed_dim = table_embeds.shape
        max_columns = column_embeds.shape[2]
        
        # Step 1: Generate column-aware table embeddings
        column_aware_table_embeds = []
        
        for i in range(max_tables):
            table_i = table_embeds[:, i:i+1, :]  # [batch_size, 1, embed_dim]
            columns_i = column_embeds[:, i, :, :]  # [batch_size, max_columns, embed_dim]
            col_mask_i = column_masks[:, i, :]  # [batch_size, max_columns]
            
            # Check if there are any valid columns for this table
            has_valid_columns = col_mask_i.any(dim=-1)  # [batch_size]
            
            if has_valid_columns.any():
                # Project embeddings
                table_i_proj = self.table_proj(table_i)
                columns_i_proj = self.column_proj(columns_i)
                
                # Apply dropout
                table_i_proj = self.dropout(table_i_proj)
                columns_i_proj = self.dropout(columns_i_proj)
                
                # Apply attention only for samples with valid columns
                attn_output, _ = self.table_column_attention(
                    query=table_i_proj,
                    key=columns_i_proj,
                    value=columns_i_proj,
                    key_padding_mask=~col_mask_i.bool()
                )
                
                # Safe residual connection and layer norm
                column_aware_table_i = self._safe_layer_norm(
                    attn_output, self.layer_norm1, table_i_proj
                )
                
                # For samples without valid columns, use original table embedding
                column_aware_table_i = torch.where(
                    has_valid_columns.unsqueeze(-1).unsqueeze(-1),
                    column_aware_table_i,
                    table_i
                )
            else:
                # No valid columns, use original table embedding
                column_aware_table_i = table_i
            
            column_aware_table_embeds.append(column_aware_table_i)
        
        # Concatenate all table embeddings
        column_aware_tables = torch.cat(column_aware_table_embeds, dim=1)  # [batch_size, max_tables, embed_dim]
        
        # Step 2: Generate schema-aware embedding
        question_proj = self.question_proj(question_embed.unsqueeze(1))  # [batch_size, 1, embed_dim]
        question_proj = self.dropout(question_proj)
        
        # Check if there are any valid tables
        has_valid_tables = table_masks.any(dim=-1)  # [batch_size]
        
        if has_valid_tables.any():
            # Apply question-table attention
            schema_aware_output, attention_weights = self.question_table_attention(
                query=question_proj,
                key=column_aware_tables,
                value=column_aware_tables,
                key_padding_mask=~table_masks.bool()
            )
            
            # Safe residual connection and layer norm
            schema_aware_embed = self._safe_layer_norm(
                schema_aware_output, self.layer_norm2, question_proj
            )
            
            # For samples without valid tables, use original question embedding
            schema_aware_embed = torch.where(
                has_valid_tables.unsqueeze(-1).unsqueeze(-1),
                schema_aware_embed,
                question_proj
            )
        else:
            # No valid tables, use original question embedding
            schema_aware_embed = question_proj
            attention_weights = None
        
        # Final projection
        schema_aware_embed = self.output_proj(schema_aware_embed.squeeze(1))
        
        # Final safety check
        if torch.isnan(schema_aware_embed).any() or torch.isinf(schema_aware_embed).any():
            print("Warning: NaN/Inf in final output, using question embedding")
            schema_aware_embed = self.output_proj(question_embed)
        
        return schema_aware_embed, attention_weights


class SchemaAwareDataset(Dataset):
    """Dataset for Schema-Aware Representation Learning with improved padding"""
    def __init__(self, embeddings_file=None, data_file=None, flag_model=None, max_tables=10, max_columns=20):
        self.max_tables = max_tables
        self.max_columns = max_columns
        
        if embeddings_file and os.path.exists(embeddings_file):
            print(f"Loading precomputed embeddings from {embeddings_file}...")
            embeddings = torch.load(embeddings_file)
            self.questions = embeddings['questions']
            self.question_embeds = embeddings['question_embeds']
            self.table_embeds = embeddings['table_embeds']
            self.column_embeds = embeddings['column_embeds']
            self.sql_embeds = embeddings['sql_embeds']
            self.table_masks = embeddings['table_masks']
            self.column_masks = embeddings['column_masks']
            self.sqls = embeddings['sqls']
        elif data_file and flag_model:
            self.flag_model = flag_model
            self._process_data(data_file)
        else:
            raise ValueError("Must provide either embeddings_file or both data_file and flag_model")
    
    def _create_safe_padding_embed(self, reference_embed):
        """Create safe padding embedding that won't cause attention issues"""
        # Use a small but non-zero vector that's orthogonal to typical embeddings
        padding_embed = torch.zeros_like(reference_embed)
        # Add a tiny constant to avoid exact zeros
        padding_embed.fill_(1e-8)
        return padding_embed
    
    def _process_data(self, data_file):
        """Process raw data and generate embeddings with better zero handling"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.questions = []
        self.question_embeds = []
        self.table_embeds = []
        self.column_embeds = []
        self.sql_embeds = []
        self.table_masks = []
        self.column_masks = []
        self.sqls = []
        
        print("Processing schema-aware dataset...")
        for item in tqdm(data, desc="Encoding questions, schemas and SQLs"):
            question = item['question']
            sql = item['query']
            
            if 'schema' in item:
                tables = item['schema']['tables']
                table_columns = item['schema']['columns']
            else:
                raise ValueError("Schema information missing from data")
            
            question_embed = torch.tensor(self.flag_model.encode(question), dtype=torch.float32)
            sql_embed = torch.tensor(self.flag_model.encode(sql), dtype=torch.float32)
            
            # Skip if embeddings contain NaN
            if torch.isnan(question_embed).any() or torch.isnan(sql_embed).any():
                print(f"Warning: NaN in embeddings for question: {question}")
                continue
            
            table_embed_list = []
            column_embed_list = []
            table_mask = torch.zeros(self.max_tables, dtype=torch.bool)
            column_mask = torch.zeros(self.max_tables, self.max_columns, dtype=torch.bool)
            
            # Process valid tables
            for i, table in enumerate(tables[:self.max_tables]):
                table_embed = torch.tensor(self.flag_model.encode(f"Table: {table}"), dtype=torch.float32)
                table_embed_list.append(table_embed)
                table_mask[i] = True
                
                columns = table_columns.get(table, [])
                table_column_embeds = []
                
                # Process valid columns
                for j, column in enumerate(columns[:self.max_columns]):
                    column_embed = torch.tensor(
                        self.flag_model.encode(f"Column: {column} in {table}"), 
                        dtype=torch.float32
                    )
                    table_column_embeds.append(column_embed)
                    column_mask[i, j] = True
                
                # Pad columns with safe embeddings
                while len(table_column_embeds) < self.max_columns:
                    padding_embed = self._create_safe_padding_embed(question_embed)
                    table_column_embeds.append(padding_embed)
                
                column_embed_list.append(torch.stack(table_column_embeds))
            
            # Pad tables with safe embeddings
            while len(table_embed_list) < self.max_tables:
                padding_table_embed = self._create_safe_padding_embed(question_embed)
                padding_column_embeds = torch.stack([
                    self._create_safe_padding_embed(question_embed) 
                    for _ in range(self.max_columns)
                ])
                table_embed_list.append(padding_table_embed)
                column_embed_list.append(padding_column_embeds)
            
            self.questions.append(question)
            self.sqls.append(sql)
            self.question_embeds.append(question_embed)
            self.sql_embeds.append(sql_embed)
            self.table_embeds.append(torch.stack(table_embed_list))
            self.column_embeds.append(torch.stack(column_embed_list))
            self.table_masks.append(table_mask)
            self.column_masks.append(column_mask)
    
    def save_embeddings(self, save_path):
        """Save precomputed embeddings"""
        embeddings = {
            'questions': self.questions,
            'question_embeds': self.question_embeds,
            'table_embeds': self.table_embeds,
            'column_embeds': self.column_embeds,
            'sql_embeds': self.sql_embeds,
            'table_masks': self.table_masks,
            'column_masks': self.column_masks,
            'sqls': self.sqls
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)
        print(f"Embeddings saved to {save_path}")
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'question_embed': self.question_embeds[idx],
            'table_embeds': self.table_embeds[idx],
            'column_embeds': self.column_embeds[idx],
            'sql_embed': self.sql_embeds[idx],
            'table_masks': self.table_masks[idx],
            'column_masks': self.column_masks[idx],
            'question': self.questions[idx],
            'sql': self.sqls[idx]
        }


class SchemaAwareTrainer:
    """Trainer for Schema-Aware Representation Learning with enhanced stability"""
    def __init__(self, 
                 model: SchemaAwareModel,
                 device: torch.device,
                 learning_rate: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Use more conservative optimizer settings
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=1e-8,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Use smoother loss function
        self.criterion = nn.SmoothL1Loss()
        
        # Gradient clipping
        self.max_grad_norm = 1.0
        
    def _safe_loss_computation(self, pred, target):
        """Safe loss computation with NaN handling"""
        # Check for NaN in predictions or targets
        if torch.isnan(pred).any() or torch.isnan(target).any():
            print("Warning: NaN detected in loss computation")
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        try:
            loss = self.criterion(pred, target)
            if torch.isnan(loss):
                print("Warning: NaN loss computed")
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            return loss
        except Exception as e:
            print(f"Error computing loss: {e}")
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with enhanced stability"""
        self.model.train()
        total_loss = 0.0
        total_similarity = 0.0
        num_valid_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            try:
                question_embeds = batch['question_embed'].to(self.device)
                table_embeds = batch['table_embeds'].to(self.device)
                column_embeds = batch['column_embeds'].to(self.device)
                sql_embeds = batch['sql_embed'].to(self.device)
                table_masks = batch['table_masks'].to(self.device)
                column_masks = batch['column_masks'].to(self.device)
                
                self.optimizer.zero_grad()
                
                schema_aware_embeds, attention_weights = self.model(
                    question_embeds, table_embeds, column_embeds, table_masks, column_masks
                )
                
                # Safe loss computation
                loss = self._safe_loss_computation(schema_aware_embeds, sql_embeds)
                
                if loss.item() == 0.0:
                    continue
                
                # Compute similarity
                with torch.no_grad():
                    similarity = F.cosine_similarity(schema_aware_embeds, sql_embeds, dim=-1).mean()
                    if torch.isnan(similarity):
                        similarity = torch.tensor(0.0)
                
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Check for problematic gradients
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: problematic gradient norm: {grad_norm}")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
                
                total_loss += loss.item()
                total_similarity += similarity.item()
                num_valid_batches += 1
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        if num_valid_batches == 0:
            return float('inf'), 0.0
        
        avg_loss = total_loss / num_valid_batches
        avg_similarity = total_similarity / num_valid_batches
        return avg_loss, avg_similarity
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model with enhanced stability"""
        self.model.eval()
        total_loss = 0.0
        total_similarity = 0.0
        num_valid_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
                try:
                    question_embeds = batch['question_embed'].to(self.device)
                    table_embeds = batch['table_embeds'].to(self.device)
                    column_embeds = batch['column_embeds'].to(self.device)
                    sql_embeds = batch['sql_embed'].to(self.device)
                    table_masks = batch['table_masks'].to(self.device)
                    column_masks = batch['column_masks'].to(self.device)
                    
                    schema_aware_embeds, _ = self.model(
                        question_embeds, table_embeds, column_embeds, table_masks, column_masks
                    )
                    
                    # Safe loss computation
                    loss = self._safe_loss_computation(schema_aware_embeds, sql_embeds)
                    
                    if loss.item() == 0.0:
                        continue
                    
                    similarity = F.cosine_similarity(schema_aware_embeds, sql_embeds, dim=-1).mean()
                    if torch.isnan(similarity):
                        similarity = torch.tensor(0.0)
                    
                    total_loss += loss.item()
                    total_similarity += similarity.item()
                    num_valid_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        if num_valid_batches == 0:
            return float('inf'), 0.0
        
        avg_loss = total_loss / num_valid_batches
        avg_similarity = total_similarity / num_valid_batches
        return avg_loss, avg_similarity


def main():
    """Main training function with enhanced stability"""
    # Configuration
    embeddings_file = './embeddings/schema_aware_embeddings.pt'
    data_file = './train_SAR_SA.json'
    model_save_path = './SAR/models/best_schema_aware_model.pth'
    flag_model_path = './plm/embeddingmodel'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create dataset
    if not os.path.exists(embeddings_file):
        print("Creating embeddings...")
        flag_model = FlagModel(flag_model_path, use_fp16=True)
        dataset = SchemaAwareDataset(data_file=data_file, flag_model=flag_model)
        dataset.save_embeddings(embeddings_file)
    else:
        print("Loading existing embeddings...")
        dataset = SchemaAwareDataset(embeddings_file=embeddings_file)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders with smaller batch size for stability
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Initialize model with smaller dimensions for stability
    model = SchemaAwareModel(
        embed_dim=1024,
        num_heads=8,
        dropout=0.1
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = SchemaAwareTrainer(model, device, learning_rate=5e-5)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_similarities = []
    val_similarities = []
    
    best_val_loss = float('inf')
    patience = 5
    no_improve_count = 0
    
    for epoch in range(20):
        print(f"\nEpoch {epoch + 1}/20")
        
        train_loss, train_sim = trainer.train_epoch(train_loader)
        val_loss, val_sim = trainer.validate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_similarities.append(train_sim)
        val_similarities.append(val_sim)
        
        print(f"Train Loss: {train_loss:.4f}, Train Similarity: {train_sim:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Similarity: {val_sim:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'similarity': val_sim
            }, model_save_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")
        else:
            no_improve_count += 1
        
        # Update learning rate and print if changed
        old_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.scheduler.step(val_loss)
        new_lr = trainer.optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        else:
            print(f"Current learning rate: {new_lr:.6f}")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
    
    print("Training completed!")


if __name__ == "__main__":
    main()