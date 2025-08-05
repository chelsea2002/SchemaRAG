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
import random
from collections import defaultdict



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
        """使用 Xavier 初始化防止梯度爆炸"""
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


class ContrastiveLearningModel(nn.Module):
    """Stage 2: Contrastive Learning Enhancement Model"""
    def __init__(self, 
                 embed_dim: int = 1024,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super(ContrastiveLearningModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Projection head, batch normalization, and gradient clipping
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),  
            nn.ReLU(),
            nn.Dropout(0.1),  # dropout
            nn.Linear(embed_dim, embed_dim // 2),  # Dimensionality reduction to improve stability
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to improve training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask to prevent bidirectional information flow"""
        mask = torch.zeros(seq_len, seq_len)
        mask[1, 0] = float('-inf')  # Schema cannot attend to question
        return mask
    
    def forward(self, 
                question_embeds: torch.Tensor, 
                schema_aware_embeds: torch.Tensor) -> torch.Tensor:
        batch_size = question_embeds.shape[0]
        
        
        combined_embeds = torch.stack([question_embeds, schema_aware_embeds], dim=1)
        causal_mask = self.create_causal_mask(2).to(combined_embeds.device)
        
        enhanced_sequence = self.transformer(combined_embeds, mask=causal_mask)
        enhanced_question = enhanced_sequence[:, 0, :]
        enhanced_question = self.layer_norm(enhanced_question)
        
        
        final_embedding = self.projection_head(enhanced_question)
        
        
        return final_embedding


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized tensors"""
    question_embeds = torch.stack([item['question_embed'] for item in batch])
    schema_aware_embeds = torch.stack([item['schema_aware_embed'] for item in batch])
    sql_embeds = torch.stack([item['sql_embed'] for item in batch])
    
    # similar samples
    max_similar = max(item['similar_question_embeds'].shape[0] for item in batch)
    embed_dim = batch[0]['similar_question_embeds'].shape[1]
    
    similar_question_embeds = []
    similar_schema_aware_embeds = []
    similar_sql_embeds = []
    
    for item in batch:
        # Get similar data for the current sample
        sim_q = item['similar_question_embeds']
        sim_s = item['similar_schema_aware_embeds']
        sim_sql = item['similar_sql_embeds']
        
        current_similar = sim_q.shape[0]
        
        if current_similar < max_similar:
            padding_needed = max_similar - current_similar
            if current_similar > 0:
                last_q = sim_q[-1:].repeat(padding_needed, 1)
                last_s = sim_s[-1:].repeat(padding_needed, 1)
                last_sql = sim_sql[-1:].repeat(padding_needed, 1)
                
                sim_q = torch.cat([sim_q, last_q], dim=0)
                sim_s = torch.cat([sim_s, last_s], dim=0)
                sim_sql = torch.cat([sim_sql, last_sql], dim=0)
            else:
                sim_q = torch.zeros(max_similar, embed_dim)
                sim_s = torch.zeros(max_similar, embed_dim)
                sim_sql = torch.zeros(max_similar, embed_dim)
        
        similar_question_embeds.append(sim_q)
        similar_schema_aware_embeds.append(sim_s)
        similar_sql_embeds.append(sim_sql)
    
    similar_question_embeds = torch.stack(similar_question_embeds)
    similar_schema_aware_embeds = torch.stack(similar_schema_aware_embeds)
    similar_sql_embeds = torch.stack(similar_sql_embeds)
    
    questions = [item['question'] for item in batch]
    sqls = [item['sql'] for item in batch]
    schemas = [item['schema'] for item in batch]
    similar_questions = [item['similar_questions'] for item in batch]
    similar_sqls = [item['similar_sqls'] for item in batch]
    similar_schemas = [item['similar_schemas'] for item in batch]
    
    return {
        'question_embed': question_embeds,
        'schema_aware_embed': schema_aware_embeds,
        'sql_embed': sql_embeds,
        'similar_question_embeds': similar_question_embeds,
        'similar_schema_aware_embeds': similar_schema_aware_embeds,
        'similar_sql_embeds': similar_sql_embeds,
        'question': questions,
        'sql': sqls,
        'schema': schemas,
        'similar_questions': similar_questions,
        'similar_sqls': similar_sqls,
        'similar_schemas': similar_schemas
    }


class ContrastiveDataset(Dataset):
    """Dataset for Stage 2 Contrastive Learning with Stage 1 model integration"""
    def __init__(self, 
                 embeddings_file: str = None,
                 data_file: str = None,
                 flag_model: FlagModel = None,
                 stage1_model_path: str = None,
                 device: torch.device = None,
                 max_tables: int = 10,
                 max_columns: int = 20,
                 max_similar: int = 3):  # Add max_similar parameter
        
        self.device = device or torch.device('cpu')
        self.stage1_model = None
        self.max_tables = max_tables
        self.max_columns = max_columns
        self.max_similar = max_similar 
        
        # Load the model trained in the first stage
        if stage1_model_path and os.path.exists(stage1_model_path):
            print(f"Loading Stage 1 model from {stage1_model_path}...")
            self.stage1_model = SchemaAwareModel(
                embed_dim=1024,
                num_heads=8,
                dropout=0.1
            )
            checkpoint = torch.load(stage1_model_path, map_location=self.device)
            self.stage1_model.load_state_dict(checkpoint['model_state_dict'])
            self.stage1_model.to(self.device)
            self.stage1_model.eval()
            print("Stage 1 model loaded successfully!")
        
        if embeddings_file and os.path.exists(embeddings_file):
            print(f"Loading precomputed embeddings from {embeddings_file}...")
            embeddings = torch.load(embeddings_file)
            self.questions = embeddings['questions']
            self.sqls = embeddings['sqls']
            self.schemas = embeddings.get('schemas', [])
            self.question_embeds = embeddings['question_embeds']
            self.sql_embeds = embeddings['sql_embeds']
            self.table_embeds = embeddings.get('table_embeds', [])
            self.column_embeds = embeddings.get('column_embeds', [])
            self.table_masks = embeddings.get('table_masks', [])
            self.column_masks = embeddings.get('column_masks', [])
            self.schema_aware_embeds = embeddings['schema_aware_embeds']
            self.similar_questions = embeddings['similar_questions']
            self.similar_sqls = embeddings['similar_sqls']
            self.similar_schemas = embeddings.get('similar_schemas', [])
            self.similar_question_embeds = embeddings['similar_question_embeds']
            self.similar_sql_embeds = embeddings['similar_sql_embeds']
            self.similar_table_embeds = embeddings.get('similar_table_embeds', [])
            self.similar_column_embeds = embeddings.get('similar_column_embeds', [])
            self.similar_table_masks = embeddings.get('similar_table_masks', [])
            self.similar_column_masks = embeddings.get('similar_column_masks', [])
            self.similar_schema_aware_embeds = embeddings['similar_schema_aware_embeds']
            
            self._normalize_similar_samples()
            
                
        elif data_file and flag_model:
            print(f"Processing data from {data_file}...")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.flag_model = flag_model
            self.questions = []
            self.sqls = []
            self.schemas = []
            self.question_embeds = []
            self.sql_embeds = []
            self.table_embeds = []
            self.column_embeds = []
            self.table_masks = []
            self.column_masks = []
            self.schema_aware_embeds = []
            self.similar_questions = []
            self.similar_sqls = []
            self.similar_schemas = []
            self.similar_question_embeds = []
            self.similar_sql_embeds = []
            self.similar_table_embeds = []
            self.similar_column_embeds = []
            self.similar_table_masks = []
            self.similar_column_masks = []
            self.similar_schema_aware_embeds = []
            
            print("Processing dataset...")
            for item in tqdm(data, desc="Encoding texts and SQLs"):
                main_question = item['question']
                main_sql = item['query']
                main_schema = item.get('schema', {})
                
                # embeddings
                main_question_embed = torch.tensor(self.flag_model.encode(main_question), dtype=torch.float32)
                main_sql_embed = torch.tensor(self.flag_model.encode(main_sql), dtype=torch.float32)
                
                # schema processing
                main_table_embeds, main_column_embeds, main_table_mask, main_column_mask = self._process_schema(main_schema, main_question_embed)
                
                self.questions.append(main_question)
                self.sqls.append(main_sql)
                self.schemas.append(main_schema)
                self.question_embeds.append(main_question_embed)
                self.sql_embeds.append(main_sql_embed)
                self.table_embeds.append(main_table_embeds)
                self.column_embeds.append(main_column_embeds)
                self.table_masks.append(main_table_mask)
                self.column_masks.append(main_column_mask)
                
                # Generate schema-aware embedding
                if self.stage1_model is not None:
                    with torch.no_grad():
                        question_batch = main_question_embed.unsqueeze(0).to(self.device)
                        table_batch = main_table_embeds.unsqueeze(0).to(self.device)
                        column_batch = main_column_embeds.unsqueeze(0).to(self.device)
                        table_mask_batch = main_table_mask.unsqueeze(0).to(self.device)
                        column_mask_batch = main_column_mask.unsqueeze(0).to(self.device)
                        
                        schema_aware_embed, _ = self.stage1_model(
                            question_batch, table_batch, column_batch, 
                            table_mask_batch, column_mask_batch
                        )
                        schema_aware_embed = schema_aware_embed.squeeze(0).cpu()
                else:
                    schema_aware_embed = main_question_embed
                
                self.schema_aware_embeds.append(schema_aware_embed)
                
                # Process similar samples
                similar_questions = []
                similar_sqls = []
                similar_schemas = []
                similar_question_embeds = []
                similar_sql_embeds = []
                similar_table_embeds = []
                similar_column_embeds = []
                similar_table_masks = []
                similar_column_masks = []
                similar_schema_aware_embeds = []
                
                for sim_item in item['similar']:
                    sim_question = sim_item['question']
                    sim_sql = sim_item['query']
                    sim_schema = sim_item.get('schema', main_schema)
                    
                    sim_question_embed = torch.tensor(self.flag_model.encode(sim_question), dtype=torch.float32)
                    sim_sql_embed = torch.tensor(self.flag_model.encode(sim_sql), dtype=torch.float32)
                    
                    # Process similar schema
                    sim_table_embeds, sim_column_embeds, sim_table_mask, sim_column_mask = self._process_schema(sim_schema, sim_question_embed)
                    
                    similar_questions.append(sim_question)
                    similar_sqls.append(sim_sql)
                    similar_schemas.append(sim_schema)
                    similar_question_embeds.append(sim_question_embed)
                    similar_sql_embeds.append(sim_sql_embed)
                    similar_table_embeds.append(sim_table_embeds)
                    similar_column_embeds.append(sim_column_embeds)
                    similar_table_masks.append(sim_table_mask)
                    similar_column_masks.append(sim_column_mask)
                    
                    # Use schema-aware embedding
                    if self.stage1_model is not None:
                        with torch.no_grad():
                            sim_question_batch = sim_question_embed.unsqueeze(0).to(self.device)
                            sim_table_batch = sim_table_embeds.unsqueeze(0).to(self.device)
                            sim_column_batch = sim_column_embeds.unsqueeze(0).to(self.device)
                            sim_table_mask_batch = sim_table_mask.unsqueeze(0).to(self.device)
                            sim_column_mask_batch = sim_column_mask.unsqueeze(0).to(self.device)
                            
                            sim_schema_aware_embed, _ = self.stage1_model(
                                sim_question_batch, sim_table_batch, sim_column_batch,
                                sim_table_mask_batch, sim_column_mask_batch
                            )
                            sim_schema_aware_embed = sim_schema_aware_embed.squeeze(0).cpu()
                    else:
                        sim_schema_aware_embed = sim_question_embed
                    
                    similar_schema_aware_embeds.append(sim_schema_aware_embed)
                
                # Padding to fixed size
                similar_questions = self._pad_or_truncate_list(similar_questions, main_question, self.max_similar)
                similar_sqls = self._pad_or_truncate_list(similar_sqls, main_sql, self.max_similar)
                similar_schemas = self._pad_or_truncate_list(similar_schemas, main_schema, self.max_similar)
                similar_question_embeds = self._pad_or_truncate_tensors(similar_question_embeds, main_question_embed, self.max_similar)
                similar_sql_embeds = self._pad_or_truncate_tensors(similar_sql_embeds, main_sql_embed, self.max_similar)
                similar_table_embeds = self._pad_or_truncate_tensors(similar_table_embeds, main_table_embeds, self.max_similar)
                similar_column_embeds = self._pad_or_truncate_tensors(similar_column_embeds, main_column_embeds, self.max_similar)
                similar_table_masks = self._pad_or_truncate_tensors(similar_table_masks, main_table_mask, self.max_similar)
                similar_column_masks = self._pad_or_truncate_tensors(similar_column_masks, main_column_mask, self.max_similar)
                similar_schema_aware_embeds = self._pad_or_truncate_tensors(similar_schema_aware_embeds, schema_aware_embed, self.max_similar)
                
                self.similar_questions.append(similar_questions)
                self.similar_sqls.append(similar_sqls)
                self.similar_schemas.append(similar_schemas)
                self.similar_question_embeds.append(torch.stack(similar_question_embeds))
                self.similar_sql_embeds.append(torch.stack(similar_sql_embeds))
                self.similar_table_embeds.append(torch.stack(similar_table_embeds))
                self.similar_column_embeds.append(torch.stack(similar_column_embeds))
                self.similar_table_masks.append(torch.stack(similar_table_masks))
                self.similar_column_masks.append(torch.stack(similar_column_masks))
                self.similar_schema_aware_embeds.append(torch.stack(similar_schema_aware_embeds))
            
        else:
            raise ValueError("Must provide either embeddings_file or both data_file and flag_model")
    
    def _create_safe_padding_embed(self, reference_embed):
        """Create safe padding embedding that won't cause attention issues"""
        # Use a small but non-zero vector that's orthogonal to typical embeddings
        padding_embed = torch.zeros_like(reference_embed)
        # Add a tiny constant to avoid exact zeros
        padding_embed.fill_(1e-8)
        return padding_embed
    
    
    def _normalize_similar_samples(self):
        print("Normalizing similar samples to fixed size...")
        
        for i in range(len(self.similar_question_embeds)):
            current_similar = self.similar_question_embeds[i]
            if current_similar.shape[0] != self.max_similar:
                # Need to adjust size
                if current_similar.shape[0] > self.max_similar:
                    # Truncate
                    self.similar_question_embeds[i] = current_similar[:self.max_similar]
                    self.similar_sql_embeds[i] = self.similar_sql_embeds[i][:self.max_similar]
                    self.similar_schema_aware_embeds[i] = self.similar_schema_aware_embeds[i][:self.max_similar]
                    if i < len(self.similar_questions):
                        self.similar_questions[i] = self.similar_questions[i][:self.max_similar]
                        self.similar_sqls[i] = self.similar_sqls[i][:self.max_similar]
                        if i < len(self.similar_schemas):
                            self.similar_schemas[i] = self.similar_schemas[i][:self.max_similar]
                else:
                    # Padding
                    padding_needed = self.max_similar - current_similar.shape[0]
                    
                    # Use the last similar sample for padding, or use the main sample
                    if current_similar.shape[0] > 0:
                        last_question = current_similar[-1:]
                        last_sql = self.similar_sql_embeds[i][-1:]
                        last_schema_aware = self.similar_schema_aware_embeds[i][-1:]
                    else:
                        # If no similar samples, use main sample
                        last_question = self.question_embeds[i].unsqueeze(0)
                        last_sql = self.sql_embeds[i].unsqueeze(0)
                        last_schema_aware = self.schema_aware_embeds[i].unsqueeze(0)
                    
                    # Padding tensors
                    for _ in range(padding_needed):
                        self.similar_question_embeds[i] = torch.cat([
                            self.similar_question_embeds[i], last_question
                        ], dim=0)
                        self.similar_sql_embeds[i] = torch.cat([
                            self.similar_sql_embeds[i], last_sql
                        ], dim=0)
                        self.similar_schema_aware_embeds[i] = torch.cat([
                            self.similar_schema_aware_embeds[i], last_schema_aware
                        ], dim=0)
                    
                    # Padding string lists
                    if i < len(self.similar_questions):
                        if len(self.similar_questions[i]) > 0:
                            last_q_str = self.similar_questions[i][-1]
                            last_s_str = self.similar_sqls[i][-1]
                        else:
                            last_q_str = self.questions[i]
                            last_s_str = self.sqls[i]
                        
                        self.similar_questions[i].extend([last_q_str] * padding_needed)
                        self.similar_sqls[i].extend([last_s_str] * padding_needed)
                        
                        if i < len(self.similar_schemas):
                            if len(self.similar_schemas[i]) > 0:
                                last_schema_str = self.similar_schemas[i][-1]
                            else:
                                last_schema_str = self.schemas[i] if i < len(self.schemas) else {}
                            self.similar_schemas[i].extend([last_schema_str] * padding_needed)
        
        print(f"All similar samples normalized to size {self.max_similar}")
    
    def _pad_or_truncate_list(self, lst, default_item, target_size):
        """Pad or truncate list to target size"""
        if len(lst) > target_size:
            return lst[:target_size]
        elif len(lst) < target_size:
            if len(lst) > 0:
                padding_item = lst[-1]
            else:
                padding_item = default_item
            return lst + [padding_item] * (target_size - len(lst))
        else:
            return lst
    
    def _pad_or_truncate_tensors(self, tensor_list, default_tensor, target_size):
        """Pad or truncate tensor list to target size"""
        if len(tensor_list) > target_size:
            return tensor_list[:target_size]
        elif len(tensor_list) < target_size:
            if len(tensor_list) > 0:
                padding_tensor = tensor_list[-1]
            else:
                padding_tensor = default_tensor
            return tensor_list + [padding_tensor] * (target_size - len(tensor_list))
        else:
            return tensor_list
    
    def _process_schema(self, schema_info, default_embed):
        """Process schema information, generate table and column embeddings"""
        if not schema_info or 'tables' not in schema_info:
            # If no schema information, return default values
            table_embeds = torch.zeros(self.max_tables, default_embed.shape[0])
            column_embeds = torch.zeros(self.max_tables, self.max_columns, default_embed.shape[0])
            table_mask = torch.zeros(self.max_tables, dtype=torch.bool)
            column_mask = torch.zeros(self.max_tables, self.max_columns, dtype=torch.bool)
            return table_embeds, column_embeds, table_mask, column_mask
        
        tables = schema_info['tables']
        table_columns = schema_info.get('columns', {})
        
        table_embed_list = []
        column_embed_list = []
        table_mask = torch.zeros(self.max_tables, dtype=torch.bool)
        column_mask = torch.zeros(self.max_tables, self.max_columns, dtype=torch.bool)
        
        for i, table in enumerate(tables[:self.max_tables]):
            table_embed = torch.tensor(self.flag_model.encode(f"Table: {table}"), dtype=torch.float32)
            table_embed_list.append(table_embed)
            table_mask[i] = True
            
            columns = table_columns.get(table, [])
            table_column_embeds = []
            
            for j, column in enumerate(columns[:self.max_columns]):
                column_embed = torch.tensor(
                    self.flag_model.encode(f"Column: {column} in {table}"), 
                    dtype=torch.float32
                )
                table_column_embeds.append(column_embed)
                column_mask[i, j] = True
            
            # Pad columns with safe embeddings
            while len(table_column_embeds) < self.max_columns:
                padding_embed = self._create_safe_padding_embed(default_embed)
                table_column_embeds.append(padding_embed)
            
            column_embed_list.append(torch.stack(table_column_embeds))
        
        # Pad tables with safe embeddings
        while len(table_embed_list) < self.max_tables:
            padding_table_embed = self._create_safe_padding_embed(default_embed)
            padding_column_embeds = torch.stack([
                self._create_safe_padding_embed(default_embed) 
                for _ in range(self.max_columns)
            ])
            table_embed_list.append(padding_table_embed)
            column_embed_list.append(padding_column_embeds)
        
        return torch.stack(table_embed_list), torch.stack(column_embed_list), table_mask, column_mask
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'question_embed': self.question_embeds[idx],
            'schema_aware_embed': self.schema_aware_embeds[idx],
            'sql_embed': self.sql_embeds[idx],
            'similar_question_embeds': self.similar_question_embeds[idx],
            'similar_schema_aware_embeds': self.similar_schema_aware_embeds[idx],
            'similar_sql_embeds': self.similar_sql_embeds[idx],
            'question': self.questions[idx],
            'sql': self.sqls[idx],
            'schema': self.schemas[idx] if hasattr(self, 'schemas') and idx < len(self.schemas) else {},
            'similar_questions': self.similar_questions[idx],
            'similar_sqls': self.similar_sqls[idx],
            'similar_schemas': self.similar_schemas[idx] if hasattr(self, 'similar_schemas') and idx < len(self.similar_schemas) else [{}] * self.max_similar
        }
    
    def save_embeddings(self, save_path):
        """Save embeddings to file"""
        embeddings = {
            'questions': self.questions,
            'sqls': self.sqls,
            'schemas': getattr(self, 'schemas', []),
            'question_embeds': self.question_embeds,
            'sql_embeds': self.sql_embeds,
            'table_embeds': getattr(self, 'table_embeds', []),
            'column_embeds': getattr(self, 'column_embeds', []),
            'table_masks': getattr(self, 'table_masks', []),
            'column_masks': getattr(self, 'column_masks', []),
            'schema_aware_embeds': self.schema_aware_embeds,
            'similar_questions': self.similar_questions,
            'similar_sqls': self.similar_sqls,
            'similar_schemas': getattr(self, 'similar_schemas', []),
            'similar_question_embeds': self.similar_question_embeds,
            'similar_sql_embeds': self.similar_sql_embeds,
            'similar_table_embeds': getattr(self, 'similar_table_embeds', []),
            'similar_column_embeds': getattr(self, 'similar_column_embeds', []),
            'similar_table_masks': getattr(self, 'similar_table_masks', []),
            'similar_column_masks': getattr(self, 'similar_column_masks', []),
            'similar_schema_aware_embeds': self.similar_schema_aware_embeds
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)
        print(f"Embeddings saved to {save_path}")


class ContrastiveTrainer:
    """Trainer for Stage 2 Contrastive Learning"""
    def __init__(self, 
                 model: ContrastiveLearningModel,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 temperature: float = 0.05,
                 similarity_weight: float = 0.5):
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        self.similarity_weight = similarity_weight
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )
        
        # Add gradient clipping parameters
        self.max_grad_norm = 1.0
        
    def contrastive_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using InfoNCE with improved numerical stability"""
        batch_size = embeddings.shape[0]
        
        # Check input
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            print("Warning: NaN or Inf detected in embeddings for contrastive loss")
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize for improved numerical stability
        embeddings = F.normalize(embeddings, dim=1, eps=1e-8)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Numerical stability handling
        similarity_matrix = torch.clamp(similarity_matrix, min=-50, max=50)
        
        # Create labels
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        # Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf detected in contrastive loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss
    
    def similarity_loss(self, 
                       main_embeddings: torch.Tensor,
                       similar_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity loss to bring similar samples closer with improved stability"""
        batch_size, num_similar, embed_dim = similar_embeddings.shape
        
        # Check inputs
        if torch.isnan(main_embeddings).any() or torch.isinf(main_embeddings).any():
            print("Warning: NaN or Inf detected in main_embeddings for similarity loss")
            main_embeddings = torch.nan_to_num(main_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(similar_embeddings).any() or torch.isinf(similar_embeddings).any():
            print("Warning: NaN or Inf detected in similar_embeddings for similarity loss")
            similar_embeddings = torch.nan_to_num(similar_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalization
        main_embeddings = F.normalize(main_embeddings, dim=1, eps=1e-8)
        similar_embeddings = F.normalize(similar_embeddings, dim=2, eps=1e-8)
        
        total_loss = 0.0
        valid_similarities = 0
        
        for i in range(num_similar):
            similarities = F.cosine_similarity(
                main_embeddings, 
                similar_embeddings[:, i, :], 
                dim=1,
                eps=1e-8
            )
            
            # Check similarity
            if not torch.isnan(similarities).any() and not torch.isinf(similarities).any():
                loss = torch.clamp(1.0 - similarities.mean(), min=0.0, max=2.0)
                total_loss += loss
                valid_similarities += 1
        
        if valid_similarities > 0:
            final_loss = total_loss / valid_similarities
        else:
            final_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Check final loss
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print("Warning: NaN or Inf detected in similarity loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return final_loss
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch with improved error handling"""
        self.model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_similarity_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training Stage 2")):
            try:
                question_embeds = batch['question_embed'].to(self.device)
                schema_aware_embeds = batch['schema_aware_embed'].to(self.device)
                similar_question_embeds = batch['similar_question_embeds'].to(self.device)
                similar_schema_aware_embeds = batch['similar_schema_aware_embeds'].to(self.device)
                
                # Check input data
                if torch.isnan(question_embeds).any() or torch.isinf(question_embeds).any():
                    print(f"Warning: NaN/Inf in question_embeds at batch {batch_idx}")
                    continue
                    
                if torch.isnan(schema_aware_embeds).any() or torch.isinf(schema_aware_embeds).any():
                    print(f"Warning: NaN/Inf in schema_aware_embeds at batch {batch_idx}")
                    continue
                
                self.optimizer.zero_grad()
                
                enhanced_embeds = self.model(question_embeds, schema_aware_embeds)
                
                # Check model output
                if torch.isnan(enhanced_embeds).any() or torch.isinf(enhanced_embeds).any():
                    print(f"Warning: NaN/Inf in model output at batch {batch_idx}")
                    continue
                
                # Process similar samples
                batch_size, max_similar, embed_dim = similar_question_embeds.shape
                similar_question_flat = similar_question_embeds.view(-1, embed_dim)
                similar_schema_flat = similar_schema_aware_embeds.view(-1, embed_dim)
                
                enhanced_similar_flat = self.model(similar_question_flat, similar_schema_flat)
                enhanced_similar = enhanced_similar_flat.view(batch_size, max_similar, embed_dim)
                
                # Compute losses
                contrastive_loss = self.contrastive_loss(enhanced_embeds)
                similarity_loss = self.similarity_loss(enhanced_embeds, enhanced_similar)
                
                # Check losses
                if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                    print(f"Warning: Invalid contrastive loss at batch {batch_idx}")
                    continue
                    
                if torch.isnan(similarity_loss) or torch.isinf(similarity_loss):
                    print(f"Warning: Invalid similarity loss at batch {batch_idx}")
                    continue
                
                total_batch_loss = contrastive_loss + self.similarity_weight * similarity_loss
                
                # Check total loss
                if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                    print(f"Warning: Invalid total loss at batch {batch_idx}")
                    continue
                
                total_batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Check gradients
                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"Warning: NaN/Inf gradients at batch {batch_idx}")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_similarity_loss += similarity_loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            return 0.0, 0.0, 0.0
        
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_similarity_loss = total_similarity_loss / num_batches
        
        return avg_loss, avg_contrastive_loss, avg_similarity_loss
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Validate the model with improved error handling"""
        self.model.eval()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_similarity_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating Stage 2")):
                try:
                    question_embeds = batch['question_embed'].to(self.device)
                    schema_aware_embeds = batch['schema_aware_embed'].to(self.device)
                    similar_question_embeds = batch['similar_question_embeds'].to(self.device)
                    similar_schema_aware_embeds = batch['similar_schema_aware_embeds'].to(self.device)
                    
                    # Check input data
                    if (torch.isnan(question_embeds).any() or torch.isinf(question_embeds).any() or
                        torch.isnan(schema_aware_embeds).any() or torch.isinf(schema_aware_embeds).any()):
                        continue
                    
                    enhanced_embeds = self.model(question_embeds, schema_aware_embeds)
                    
                    # Check model output
                    if torch.isnan(enhanced_embeds).any() or torch.isinf(enhanced_embeds).any():
                        continue
                    
                    batch_size, max_similar, embed_dim = similar_question_embeds.shape
                    similar_question_flat = similar_question_embeds.view(-1, embed_dim)
                    similar_schema_flat = similar_schema_aware_embeds.view(-1, embed_dim)
                    
                    enhanced_similar_flat = self.model(similar_question_flat, similar_schema_flat)
                    enhanced_similar = enhanced_similar_flat.view(batch_size, max_similar, embed_dim)
                    
                    contrastive_loss = self.contrastive_loss(enhanced_embeds)
                    similarity_loss = self.similarity_loss(enhanced_embeds, enhanced_similar)
                    
                    if (not torch.isnan(contrastive_loss) and not torch.isinf(contrastive_loss) and
                        not torch.isnan(similarity_loss) and not torch.isinf(similarity_loss)):
                        
                        total_batch_loss = contrastive_loss + self.similarity_weight * similarity_loss
                        
                        total_loss += total_batch_loss.item()
                        total_contrastive_loss += contrastive_loss.item()
                        total_similarity_loss += similarity_loss.item()
                        num_batches += 1
                
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        if num_batches == 0:
            return float('inf'), float('inf'), float('inf')
        
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_similarity_loss = total_similarity_loss / num_batches
        
        return avg_loss, avg_contrastive_loss, avg_similarity_loss


def train_stage2_contrastive_model(data_file: str = None,
                                 embeddings_file: str = None,
                                 stage1_model_path: str = None,
                                 flag_model_path: str = None,
                                 num_epochs: int = 15,
                                 batch_size: int = 32,
                                 learning_rate: float = 1e-4,
                                 temperature: float = 0.05,
                                 device: str = 'cuda',
                                 save_path: str = './SAR/models/stage2_contrastive_model.pth',
                                 output_dir: str = './training_plots/stage2'):
    """Train Stage 2 Contrastive Learning Model with improved stability"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if embeddings_file and os.path.exists(embeddings_file):
        dataset = ContrastiveDataset(
            embeddings_file=embeddings_file,
            stage1_model_path=stage1_model_path,
            device=device,
            max_similar=3
        )
    elif data_file and flag_model_path:
        flag_model = FlagModel(flag_model_path, use_fp16=True)
        dataset = ContrastiveDataset(
            data_file=data_file,
            flag_model=flag_model,
            stage1_model_path=stage1_model_path,
            device=device,
            max_similar=3
        )
        
        if embeddings_file:
            dataset.save_embeddings(embeddings_file)
    else:
        raise ValueError("Must provide either embeddings_file or both data_file and flag_model_path")
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Use smaller batch size for improved stability
    effective_batch_size = min(batch_size, 8)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=effective_batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    model = ContrastiveLearningModel(
        embed_dim=1024,
        num_layers=2, 
        num_heads=8,
        dropout=0.1
    )
    
    print(f"Stage 2 model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    

    trainer = ContrastiveTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate * 0.1, 
        temperature=max(temperature, 0.1),  
        similarity_weight=0.3 
    )
    
    train_losses = []
    val_losses = []
    train_contrastive_losses = []
    val_contrastive_losses = []
    train_similarity_losses = []
    val_similarity_losses = []
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_cont_loss, train_sim_loss = trainer.train_epoch(train_loader)
        
        if math.isnan(train_loss) or math.isinf(train_loss):
            print(f"Invalid training loss at epoch {epoch + 1}, stopping training")
            break
        
        train_losses.append(train_loss)
        train_contrastive_losses.append(train_cont_loss)
        train_similarity_losses.append(train_sim_loss)
        
        val_loss, val_cont_loss, val_sim_loss = trainer.validate(val_loader)
        
        if math.isnan(val_loss) or math.isinf(val_loss):
            print(f"Invalid validation loss at epoch {epoch + 1}")
            val_loss = float('inf')
        
        val_losses.append(val_loss)
        val_contrastive_losses.append(val_cont_loss)
        val_similarity_losses.append(val_sim_loss)
        
        print(f"Train Loss: {train_loss:.4f} (Contrastive: {train_cont_loss:.4f}, Similarity: {train_sim_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Contrastive: {val_cont_loss:.4f}, Similarity: {val_sim_loss:.4f})")
        print(f"LR: {trainer.scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'contrastive_loss': val_cont_loss,
                'similarity_loss': val_sim_loss
            }, save_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        trainer.scheduler.step()
    
    # Save final model
    final_save_path = save_path.replace('.pth', '_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'epoch': len(train_losses) - 1,
        'loss': val_losses[-1] if val_losses else float('inf'),
        'contrastive_loss': val_contrastive_losses[-1] if val_contrastive_losses else float('inf'),
        'similarity_loss': val_similarity_losses[-1] if val_similarity_losses else float('inf')
    }, final_save_path)
    
    # Plot training charts
    if train_losses:  # Only plot if there's valid data
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        epochs_range = range(1, len(train_losses) + 1)
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Stage 2 Total Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, train_contrastive_losses, label='Train Contrastive')
        plt.plot(epochs_range, val_contrastive_losses, label='Val Contrastive')
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss')
        plt.title('Contrastive Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, train_similarity_losses, label='Train Similarity')
        plt.plot(epochs_range, val_similarity_losses, label='Val Similarity')
        plt.xlabel('Epoch')
        plt.ylabel('Similarity Loss')
        plt.title('Similarity Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stage2_training_metrics.png'))
        plt.close()
        
        # Separate loss charts
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss.png'))
        plt.close()
    
    print("Stage 2 training completed!")
    print(f"Best model saved at: {save_path}")
    print(f"Final model saved at: {final_save_path}")
    print(f"Training plots saved in: {output_dir}")


def main_stage2():
    """Main function for Stage 2 training with improved parameters"""
    data_file = './train_SAR_CL.json'
    embeddings_file = 'embeddings/stage2_embeddings.pt'
    stage1_model_path = './SAR/models/best_schema_aware_model.pth'
    flag_model_path = '../plm/embeddingmodel'
    stage2_model_save_path = './SAR/models/best_contrastive_model.pth'
    output_dir = './training_plots/stage2'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_stage2_contrastive_model(
        data_file=data_file,
        embeddings_file=embeddings_file,
        stage1_model_path=stage1_model_path,
        flag_model_path=flag_model_path,
        num_epochs=20,
        batch_size=8,  
        learning_rate=5e-5,  
        temperature=0.1,
        device=device,
        save_path=stage2_model_save_path,
        output_dir=output_dir
    )
    
    print("Stage 2 training completed!")


if __name__ == "__main__":

    main_stage2()
