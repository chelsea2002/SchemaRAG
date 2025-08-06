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
import argparse
import logging
from datetime import datetime


# Setup logging
def setup_logging(log_dir: str = "./logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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
            all_masked = key_padding_mask.all(dim=-1)
            
            if all_masked.any():
                attn_output = torch.zeros_like(query)
                attn_weights = None
                
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
                        attn_output = query.clone()
                
                return attn_output, attn_weights
        
        try:
            return self.attention(query, key, value, 
                                key_padding_mask=key_padding_mask,
                                attn_mask=attn_mask)
        except Exception as e:
            print(f"Error in attention computation: {e}")
            return query.clone(), None


class SchemaAwareModel(nn.Module):
    """Stage 1: Schema-Aware Representation Learning Model"""
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super(SchemaAwareModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.table_column_attention = SafeMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        
        self.question_table_attention = SafeMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.table_proj = nn.Linear(embed_dim, embed_dim)
        self.column_proj = nn.Linear(embed_dim, embed_dim)
        self.question_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
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
            
            if torch.isnan(x).any() or torch.isinf(x).any():
                return residual if residual is not None else torch.zeros_like(x)
            
            output = layer_norm(x)
            
            if torch.isnan(output).any() or torch.isinf(output).any():
                return residual if residual is not None else torch.zeros_like(x)
            
            return output
            
        except Exception as e:
            print(f"Error in layer norm: {e}")
            return residual if residual is not None else torch.zeros_like(x)
    
    def forward(self, question_embed, table_embeds, column_embeds, table_masks, column_masks):
        """Forward pass for schema-aware model"""
        batch_size, max_tables, embed_dim = table_embeds.shape
        max_columns = column_embeds.shape[2]
        
        # Step 1: Generate column-aware table embeddings
        column_aware_table_embeds = []
        
        for i in range(max_tables):
            table_i = table_embeds[:, i:i+1, :]
            columns_i = column_embeds[:, i, :, :]
            col_mask_i = column_masks[:, i, :]
            
            has_valid_columns = col_mask_i.any(dim=-1)
            
            if has_valid_columns.any():
                table_i_proj = self.table_proj(table_i)
                columns_i_proj = self.column_proj(columns_i)
                
                table_i_proj = self.dropout(table_i_proj)
                columns_i_proj = self.dropout(columns_i_proj)
                
                attn_output, _ = self.table_column_attention(
                    query=table_i_proj,
                    key=columns_i_proj,
                    value=columns_i_proj,
                    key_padding_mask=~col_mask_i.bool()
                )
                
                column_aware_table_i = self._safe_layer_norm(
                    attn_output, self.layer_norm1, table_i_proj
                )
                
                column_aware_table_i = torch.where(
                    has_valid_columns.unsqueeze(-1).unsqueeze(-1),
                    column_aware_table_i,
                    table_i
                )
            else:
                column_aware_table_i = table_i
            
            column_aware_table_embeds.append(column_aware_table_i)
        
        column_aware_tables = torch.cat(column_aware_table_embeds, dim=1)
        
        # Step 2: Generate schema-aware embedding
        question_proj = self.question_proj(question_embed.unsqueeze(1))
        question_proj = self.dropout(question_proj)
        
        has_valid_tables = table_masks.any(dim=-1)
        
        if has_valid_tables.any():
            schema_aware_output, attention_weights = self.question_table_attention(
                query=question_proj,
                key=column_aware_tables,
                value=column_aware_tables,
                key_padding_mask=~table_masks.bool()
            )
            
            schema_aware_embed = self._safe_layer_norm(
                schema_aware_output, self.layer_norm2, question_proj
            )
            
            schema_aware_embed = torch.where(
                has_valid_tables.unsqueeze(-1).unsqueeze(-1),
                schema_aware_embed,
                question_proj
            )
        else:
            schema_aware_embed = question_proj
            attention_weights = None
        
        schema_aware_embed = self.output_proj(schema_aware_embed.squeeze(1))
        
        if torch.isnan(schema_aware_embed).any() or torch.isinf(schema_aware_embed).any():
            schema_aware_embed = self.output_proj(question_embed)
        
        return schema_aware_embed, attention_weights


class ContrastiveLearningModel(nn.Module):
    """Stage 2: Contrastive Learning Enhancement Model"""
    def __init__(self, embed_dim: int = 1024, num_layers: int = 3, num_heads: int = 8, dropout: float = 0.1):
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
        
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
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
        mask[1, 0] = float('-inf')
        return mask
    
    def forward(self, question_embeds, schema_aware_embeds):
        batch_size = question_embeds.shape[0]
        
        combined_embeds = torch.stack([question_embeds, schema_aware_embeds], dim=1)
        causal_mask = self.create_causal_mask(2).to(combined_embeds.device)
        
        enhanced_sequence = self.transformer(combined_embeds, mask=causal_mask)
        enhanced_question = enhanced_sequence[:, 0, :]
        enhanced_question = self.layer_norm(enhanced_question)
        
        final_embedding = self.projection_head(enhanced_question)
        
        return final_embedding


class UnifiedDataset(Dataset):
    """Unified dataset that can handle both Stage 1 and Stage 2 training"""
    def __init__(self, 
                 stage1_data_file: str = None,
                 stage2_data_file: str = None,
                 embeddings_file: str = None,
                 flag_model: FlagModel = None,
                 stage1_model: SchemaAwareModel = None,
                 device: torch.device = None,
                 max_tables: int = 10,
                 max_columns: int = 20,
                 max_similar: int = 3,
                 stage: int = 1):
        
        self.device = device or torch.device('cpu')
        self.stage1_model = stage1_model
        self.max_tables = max_tables
        self.max_columns = max_columns
        self.max_similar = max_similar
        self.stage = stage
        
        if embeddings_file and os.path.exists(embeddings_file):
            self._load_embeddings(embeddings_file)
        else:
            if stage == 1 and stage1_data_file:
                self._process_stage1_data(stage1_data_file, flag_model)
            elif stage == 2 and stage2_data_file:
                self._process_stage2_data(stage2_data_file, flag_model)
            else:
                raise ValueError(f"Invalid configuration for stage {stage}")
    
    def _create_safe_padding_embed(self, reference_embed):
        """Create safe padding embedding"""
        padding_embed = torch.zeros_like(reference_embed)
        padding_embed.fill_(1e-8)
        return padding_embed
    
    def _process_schema(self, schema_info, default_embed):
        """Process schema information and generate embeddings"""
        if not schema_info or 'tables' not in schema_info:
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
            
            while len(table_column_embeds) < self.max_columns:
                padding_embed = self._create_safe_padding_embed(default_embed)
                table_column_embeds.append(padding_embed)
            
            column_embed_list.append(torch.stack(table_column_embeds))
        
        while len(table_embed_list) < self.max_tables:
            padding_table_embed = self._create_safe_padding_embed(default_embed)
            padding_column_embeds = torch.stack([
                self._create_safe_padding_embed(default_embed) 
                for _ in range(self.max_columns)
            ])
            table_embed_list.append(padding_table_embed)
            column_embed_list.append(padding_column_embeds)
        
        return torch.stack(table_embed_list), torch.stack(column_embed_list), table_mask, column_mask
    
    def _process_stage1_data(self, data_file, flag_model):
        """Process data for Stage 1 training"""
        self.flag_model = flag_model
        
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
        self.schemas = []
        
        print("Processing Stage 1 dataset...")
        for item in tqdm(data, desc="Encoding Stage 1 data"):
            question = item['question']
            sql = item['query']
            schema = item.get('schema', {})
            
            question_embed = torch.tensor(flag_model.encode(question), dtype=torch.float32)
            sql_embed = torch.tensor(flag_model.encode(sql), dtype=torch.float32)
            
            if torch.isnan(question_embed).any() or torch.isnan(sql_embed).any():
                continue
            
            table_embeds, column_embeds, table_mask, column_mask = self._process_schema(schema, question_embed)
            
            self.questions.append(question)
            self.sqls.append(sql)
            self.schemas.append(schema)
            self.question_embeds.append(question_embed)
            self.sql_embeds.append(sql_embed)
            self.table_embeds.append(table_embeds)
            self.column_embeds.append(column_embeds)
            self.table_masks.append(table_mask)
            self.column_masks.append(column_mask)
    
    def _process_stage2_data(self, data_file, flag_model):
        """Process data for Stage 2 training"""
        self.flag_model = flag_model
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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
        self.similar_schema_aware_embeds = []
        
        print("Processing Stage 2 dataset...")
        for item in tqdm(data, desc="Encoding Stage 2 data"):
            main_question = item['question']
            main_sql = item['query']
            main_schema = item.get('schema', {})
            
            main_question_embed = torch.tensor(flag_model.encode(main_question), dtype=torch.float32)
            main_sql_embed = torch.tensor(flag_model.encode(main_sql), dtype=torch.float32)
            
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
            
            # Generate schema-aware embedding using Stage 1 model
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
            similar_schema_aware_embeds = []
            
            for sim_item in item.get('similar', []):
                sim_question = sim_item['question']
                sim_sql = sim_item['query']
                sim_schema = sim_item.get('schema', main_schema)
                
                sim_question_embed = torch.tensor(flag_model.encode(sim_question), dtype=torch.float32)
                sim_sql_embed = torch.tensor(flag_model.encode(sim_sql), dtype=torch.float32)
                
                similar_questions.append(sim_question)
                similar_sqls.append(sim_sql)
                similar_schemas.append(sim_schema)
                similar_question_embeds.append(sim_question_embed)
                similar_sql_embeds.append(sim_sql_embed)
                
                # Generate schema-aware embedding for similar sample
                if self.stage1_model is not None:
                    sim_table_embeds, sim_column_embeds, sim_table_mask, sim_column_mask = self._process_schema(sim_schema, sim_question_embed)
                    
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
            
            # Pad/truncate to max_similar
            similar_questions = self._pad_or_truncate_list(similar_questions, main_question, self.max_similar)
            similar_sqls = self._pad_or_truncate_list(similar_sqls, main_sql, self.max_similar)
            similar_schemas = self._pad_or_truncate_list(similar_schemas, main_schema, self.max_similar)
            similar_question_embeds = self._pad_or_truncate_tensors(similar_question_embeds, main_question_embed, self.max_similar)
            similar_sql_embeds = self._pad_or_truncate_tensors(similar_sql_embeds, main_sql_embed, self.max_similar)
            similar_schema_aware_embeds = self._pad_or_truncate_tensors(similar_schema_aware_embeds, schema_aware_embed, self.max_similar)
            
            self.similar_questions.append(similar_questions)
            self.similar_sqls.append(similar_sqls)
            self.similar_schemas.append(similar_schemas)
            self.similar_question_embeds.append(torch.stack(similar_question_embeds))
            self.similar_sql_embeds.append(torch.stack(similar_sql_embeds))
            self.similar_schema_aware_embeds.append(torch.stack(similar_schema_aware_embeds))
    
    def _pad_or_truncate_list(self, lst, default_item, target_size):
        """Pad or truncate list to target size"""
        if len(lst) > target_size:
            return lst[:target_size]
        elif len(lst) < target_size:
            padding_item = lst[-1] if lst else default_item
            return lst + [padding_item] * (target_size - len(lst))
        return lst
    
    def _pad_or_truncate_tensors(self, tensor_list, default_tensor, target_size):
        """Pad or truncate tensor list to target size"""
        if len(tensor_list) > target_size:
            return tensor_list[:target_size]
        elif len(tensor_list) < target_size:
            padding_tensor = tensor_list[-1] if tensor_list else default_tensor
            return tensor_list + [padding_tensor] * (target_size - len(tensor_list))
        return tensor_list
    
    def _load_embeddings(self, embeddings_file):
        """Load precomputed embeddings"""
        embeddings = torch.load(embeddings_file)
        for key, value in embeddings.items():
            setattr(self, key, value)
    
    def save_embeddings(self, save_path):
        """Save embeddings to file"""
        embeddings = {}
        for attr in ['questions', 'sqls', 'schemas', 'question_embeds', 'sql_embeds', 
                     'table_embeds', 'column_embeds', 'table_masks', 'column_masks']:
            if hasattr(self, attr):
                embeddings[attr] = getattr(self, attr)
        
        # Stage 2 specific attributes
        if self.stage == 2:
            for attr in ['schema_aware_embeds', 'similar_questions', 'similar_sqls', 
                        'similar_schemas', 'similar_question_embeds', 'similar_sql_embeds',
                        'similar_schema_aware_embeds']:
                if hasattr(self, attr):
                    embeddings[attr] = getattr(self, attr)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)
        print(f"Embeddings saved to {save_path}")
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        if self.stage == 1:
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
        else:  # Stage 2
            return {
                'question_embed': self.question_embeds[idx],
                'schema_aware_embed': self.schema_aware_embeds[idx],
                'sql_embed': self.sql_embeds[idx],
                'similar_question_embeds': self.similar_question_embeds[idx],
                'similar_schema_aware_embeds': self.similar_schema_aware_embeds[idx],
                'similar_sql_embeds': self.similar_sql_embeds[idx],
                'question': self.questions[idx],
                'sql': self.sqls[idx],
                'schema': self.schemas[idx],
                'similar_questions': self.similar_questions[idx],
                'similar_sqls': self.similar_sqls[idx],
                'similar_schemas': self.similar_schemas[idx]
            }


class UnifiedTrainer:
    """Unified trainer for both Stage 1 and Stage 2"""
    def __init__(self, 
                 stage1_model: SchemaAwareModel = None,
                 stage2_model: ContrastiveLearningModel = None,
                 device: torch.device = None,
                 stage1_lr: float = 5e-5,
                 stage2_lr: float = 1e-4,
                 temperature: float = 0.1,
                 similarity_weight: float = 0.3):
        
        self.device = device or torch.device('cpu')
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.temperature = temperature
        self.similarity_weight = similarity_weight
        
        # Stage 1 optimizer and scheduler
        if stage1_model:
            self.stage1_optimizer = optim.AdamW(
                stage1_model.parameters(), 
                lr=stage1_lr, 
                eps=1e-8,
                weight_decay=1e-5
            )
            self.stage1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.stage1_optimizer, mode='min', factor=0.5, patience=3
            )
            self.stage1_criterion = nn.SmoothL1Loss()
        
        # Stage 2 optimizer and scheduler
        if stage2_model:
            self.stage2_optimizer = optim.Adam(
                stage2_model.parameters(), 
                lr=stage2_lr, 
                weight_decay=1e-5
            )
            self.stage2_scheduler = torch.optim.lr_scheduler.StepLR(
                self.stage2_optimizer, step_size=10, gamma=0.1
            )
        
        self.max_grad_norm = 1.0
    
    def _safe_loss_computation(self, pred, target, criterion):
        """Safe loss computation with NaN handling"""
        if torch.isnan(pred).any() or torch.isnan(target).any():
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        try:
            loss = criterion(pred, target)
            if torch.isnan(loss):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            return loss
        except Exception as e:
            print(f"Error computing loss: {e}")
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
    
    def contrastive_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using InfoNCE"""
        batch_size = embeddings.shape[0]
        
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        embeddings = F.normalize(embeddings, dim=1, eps=1e-8)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, min=-50, max=50)
        
        labels = torch.arange(batch_size, device=self.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss
    
    def similarity_loss(self, main_embeddings: torch.Tensor, similar_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity loss for Stage 2"""
        batch_size, num_similar, embed_dim = similar_embeddings.shape
        
        if torch.isnan(main_embeddings).any() or torch.isinf(main_embeddings).any():
            main_embeddings = torch.nan_to_num(main_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(similar_embeddings).any() or torch.isinf(similar_embeddings).any():
            similar_embeddings = torch.nan_to_num(similar_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
            
            if not torch.isnan(similarities).any() and not torch.isinf(similarities).any():
                loss = torch.clamp(1.0 - similarities.mean(), min=0.0, max=2.0)
                total_loss += loss
                valid_similarities += 1
        
        if valid_similarities > 0:
            final_loss = total_loss / valid_similarities
        else:
            final_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return final_loss
    
    def train_stage1_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train Stage 1 for one epoch"""
        if not self.stage1_model:
            return 0.0, 0.0
            
        self.stage1_model.train()
        total_loss = 0.0
        total_similarity = 0.0
        num_valid_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training Stage 1")):
            try:
                question_embeds = batch['question_embed'].to(self.device)
                table_embeds = batch['table_embeds'].to(self.device)
                column_embeds = batch['column_embeds'].to(self.device)
                sql_embeds = batch['sql_embed'].to(self.device)
                table_masks = batch['table_masks'].to(self.device)
                column_masks = batch['column_masks'].to(self.device)
                
                self.stage1_optimizer.zero_grad()
                
                schema_aware_embeds, _ = self.stage1_model(
                    question_embeds, table_embeds, column_embeds, table_masks, column_masks
                )
                
                loss = self._safe_loss_computation(schema_aware_embeds, sql_embeds, self.stage1_criterion)
                
                if loss.item() == 0.0:
                    continue
                
                with torch.no_grad():
                    similarity = F.cosine_similarity(schema_aware_embeds, sql_embeds, dim=-1).mean()
                    if torch.isnan(similarity):
                        similarity = torch.tensor(0.0)
                
                loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(self.stage1_model.parameters(), self.max_grad_norm)
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.stage1_optimizer.zero_grad()
                    continue
                
                self.stage1_optimizer.step()
                
                total_loss += loss.item()
                total_similarity += similarity.item()
                num_valid_batches += 1
                
            except Exception as e:
                print(f"Error in Stage 1 training batch {batch_idx}: {e}")
                continue
        
        if num_valid_batches == 0:
            return float('inf'), 0.0
        
        return total_loss / num_valid_batches, total_similarity / num_valid_batches
    
    def train_stage2_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Train Stage 2 for one epoch"""
        if not self.stage2_model:
            return 0.0, 0.0, 0.0
            
        self.stage2_model.train()
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
                
                if (torch.isnan(question_embeds).any() or torch.isinf(question_embeds).any() or
                    torch.isnan(schema_aware_embeds).any() or torch.isinf(schema_aware_embeds).any()):
                    continue
                
                self.stage2_optimizer.zero_grad()
                
                enhanced_embeds = self.stage2_model(question_embeds, schema_aware_embeds)
                
                if torch.isnan(enhanced_embeds).any() or torch.isinf(enhanced_embeds).any():
                    continue
                
                batch_size, max_similar, embed_dim = similar_question_embeds.shape
                similar_question_flat = similar_question_embeds.view(-1, embed_dim)
                similar_schema_flat = similar_schema_aware_embeds.view(-1, embed_dim)
                
                enhanced_similar_flat = self.stage2_model(similar_question_flat, similar_schema_flat)
                enhanced_similar = enhanced_similar_flat.view(batch_size, max_similar, embed_dim)
                
                contrastive_loss = self.contrastive_loss(enhanced_embeds)
                similarity_loss = self.similarity_loss(enhanced_embeds, enhanced_similar)
                
                if (torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss) or
                    torch.isnan(similarity_loss) or torch.isinf(similarity_loss)):
                    continue
                
                total_batch_loss = contrastive_loss + self.similarity_weight * similarity_loss
                
                if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                    continue
                
                total_batch_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.stage2_model.parameters(), self.max_grad_norm)
                
                has_nan_grad = False
                for param in self.stage2_model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    self.stage2_optimizer.zero_grad()
                    continue
                
                self.stage2_optimizer.step()
                
                total_loss += total_batch_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_similarity_loss += similarity_loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in Stage 2 training batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            return 0.0, 0.0, 0.0
        
        return total_loss / num_batches, total_contrastive_loss / num_batches, total_similarity_loss / num_batches
    
    def validate_stage1(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate Stage 1"""
        if not self.stage1_model:
            return float('inf'), 0.0
            
        self.stage1_model.eval()
        total_loss = 0.0
        total_similarity = 0.0
        num_valid_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating Stage 1")):
                try:
                    question_embeds = batch['question_embed'].to(self.device)
                    table_embeds = batch['table_embeds'].to(self.device)
                    column_embeds = batch['column_embeds'].to(self.device)
                    sql_embeds = batch['sql_embed'].to(self.device)
                    table_masks = batch['table_masks'].to(self.device)
                    column_masks = batch['column_masks'].to(self.device)
                    
                    schema_aware_embeds, _ = self.stage1_model(
                        question_embeds, table_embeds, column_embeds, table_masks, column_masks
                    )
                    
                    loss = self._safe_loss_computation(schema_aware_embeds, sql_embeds, self.stage1_criterion)
                    
                    if loss.item() == 0.0:
                        continue
                    
                    similarity = F.cosine_similarity(schema_aware_embeds, sql_embeds, dim=-1).mean()
                    if torch.isnan(similarity):
                        similarity = torch.tensor(0.0)
                    
                    total_loss += loss.item()
                    total_similarity += similarity.item()
                    num_valid_batches += 1
                    
                except Exception as e:
                    print(f"Error in Stage 1 validation batch {batch_idx}: {e}")
                    continue
        
        if num_valid_batches == 0:
            return float('inf'), 0.0
        
        return total_loss / num_valid_batches, total_similarity / num_valid_batches
    
    def validate_stage2(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Validate Stage 2"""
        if not self.stage2_model:
            return float('inf'), float('inf'), float('inf')
            
        self.stage2_model.eval()
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
                    
                    if (torch.isnan(question_embeds).any() or torch.isinf(question_embeds).any() or
                        torch.isnan(schema_aware_embeds).any() or torch.isinf(schema_aware_embeds).any()):
                        continue
                    
                    enhanced_embeds = self.stage2_model(question_embeds, schema_aware_embeds)
                    
                    if torch.isnan(enhanced_embeds).any() or torch.isinf(enhanced_embeds).any():
                        continue
                    
                    batch_size, max_similar, embed_dim = similar_question_embeds.shape
                    similar_question_flat = similar_question_embeds.view(-1, embed_dim)
                    similar_schema_flat = similar_schema_aware_embeds.view(-1, embed_dim)
                    
                    enhanced_similar_flat = self.stage2_model(similar_question_flat, similar_schema_flat)
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
                    print(f"Error in Stage 2 validation batch {batch_idx}: {e}")
                    continue
        
        if num_batches == 0:
            return float('inf'), float('inf'), float('inf')
        
        return total_loss / num_batches, total_contrastive_loss / num_batches, total_similarity_loss / num_batches


def custom_collate_fn_stage2(batch):
    """Custom collate function for Stage 2"""
    question_embeds = torch.stack([item['question_embed'] for item in batch])
    schema_aware_embeds = torch.stack([item['schema_aware_embed'] for item in batch])
    sql_embeds = torch.stack([item['sql_embed'] for item in batch])
    
    max_similar = max(item['similar_question_embeds'].shape[0] for item in batch)
    embed_dim = batch[0]['similar_question_embeds'].shape[1]
    
    similar_question_embeds = []
    similar_schema_aware_embeds = []
    similar_sql_embeds = []
    
    for item in batch:
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


def plot_training_curves(stage1_metrics, stage2_metrics, output_dir):
    """Plot training curves for both stages"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Stage 1 plots
    if stage1_metrics['train_losses']:
        epochs1 = range(1, len(stage1_metrics['train_losses']) + 1)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs1, stage1_metrics['train_losses'], label='Train Loss', color='blue')
        plt.plot(epochs1, stage1_metrics['val_losses'], label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Stage 1: Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs1, stage1_metrics['train_similarities'], label='Train Similarity', color='blue')
        plt.plot(epochs1, stage1_metrics['val_similarities'], label='Val Similarity', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Similarity')
        plt.title('Stage 1: Cosine Similarity')
        plt.legend()
        plt.grid(True)
    
    # Stage 2 plots
    if stage2_metrics['train_losses']:
        epochs2 = range(1, len(stage2_metrics['train_losses']) + 1)
        
        plt.subplot(2, 2, 3)
        plt.plot(epochs2, stage2_metrics['train_losses'], label='Train Total', color='blue')
        plt.plot(epochs2, stage2_metrics['val_losses'], label='Val Total', color='red')
        plt.plot(epochs2, stage2_metrics['train_contrastive'], label='Train Contrastive', color='green', linestyle='--')
        plt.plot(epochs2, stage2_metrics['train_similarity'], label='Train Similarity', color='orange', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Stage 2: Losses')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(epochs2, stage2_metrics['val_contrastive'], label='Val Contrastive', color='green')
        plt.plot(epochs2, stage2_metrics['val_similarity'], label='Val Similarity', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Stage 2: Validation Losses')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Separate detailed plots for each stage
    if stage1_metrics['train_losses']:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs1, stage1_metrics['train_losses'], label='Train Loss', linewidth=2)
        plt.plot(epochs1, stage1_metrics['val_losses'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Stage 1: Schema-Aware Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'stage1_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    if stage2_metrics['train_losses']:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs2, stage2_metrics['train_losses'], label='Train Total Loss', linewidth=2)
        plt.plot(epochs2, stage2_metrics['val_losses'], label='Validation Total Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Stage 2: Contrastive Learning Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'stage2_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()


def train_unified_pipeline(config):
    """Main training pipeline for both stages"""
    logger = setup_logging(config.log_dir)
    logger.info("Starting unified training pipeline")
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Load flag model
    logger.info(f"Loading embedding model from {config.flag_model_path}")
    flag_model = FlagModel(config.flag_model_path, use_fp16=True)
    
    # Initialize models
    stage1_model = SchemaAwareModel(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        dropout=config.dropout
    ).to(device)
    
    stage2_model = ContrastiveLearningModel(
        embed_dim=config.embed_dim,
        num_layers=config.stage2_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    ).to(device)
    
    logger.info(f"Stage 1 model: {sum(p.numel() for p in stage1_model.parameters())} parameters")
    logger.info(f"Stage 2 model: {sum(p.numel() for p in stage2_model.parameters())} parameters")
    
    # Initialize trainer
    trainer = UnifiedTrainer(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        device=device,
        stage1_lr=config.stage1_lr,
        stage2_lr=config.stage2_lr,
        temperature=config.temperature,
        similarity_weight=config.similarity_weight
    )
    
    # Metrics tracking
    stage1_metrics = {
        'train_losses': [], 'val_losses': [],
        'train_similarities': [], 'val_similarities': []
    }
    stage2_metrics = {
        'train_losses': [], 'val_losses': [],
        'train_contrastive': [], 'val_contrastive': [],
        'train_similarity': [], 'val_similarity': []
    }
    
    # STAGE 1: Schema-Aware Training
    logger.info("="*50)
    logger.info("STAGE 1: Schema-Aware Representation Learning")
    logger.info("="*50)
    
    # Load or create Stage 1 dataset
    if os.path.exists(config.stage1_embeddings):
        logger.info(f"Loading Stage 1 embeddings from {config.stage1_embeddings}")
        stage1_dataset = UnifiedDataset(embeddings_file=config.stage1_embeddings, stage=1)
    else:
        logger.info(f"Creating Stage 1 dataset from {config.stage1_data}")
        stage1_dataset = UnifiedDataset(
            stage1_data_file=config.stage1_data,
            flag_model=flag_model,
            max_tables=config.max_tables,
            max_columns=config.max_columns,
            stage=1
        )
        stage1_dataset.save_embeddings(config.stage1_embeddings)
    
    # Split Stage 1 dataset
    train_size = int(0.9 * len(stage1_dataset))
    val_size = len(stage1_dataset) - train_size
    stage1_train, stage1_val = torch.utils.data.random_split(
        stage1_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    stage1_train_loader = DataLoader(
        stage1_train, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    stage1_val_loader = DataLoader(
        stage1_val, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    
    # Train Stage 1
    best_stage1_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.stage1_epochs):
        logger.info(f"\nStage 1 - Epoch {epoch + 1}/{config.stage1_epochs}")
        
        train_loss, train_sim = trainer.train_stage1_epoch(stage1_train_loader)
        val_loss, val_sim = trainer.validate_stage1(stage1_val_loader)
        
        stage1_metrics['train_losses'].append(train_loss)
        stage1_metrics['val_losses'].append(val_loss)
        stage1_metrics['train_similarities'].append(train_sim)
        stage1_metrics['val_similarities'].append(val_sim)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Similarity: {train_sim:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Similarity: {val_sim:.4f}")
        
        # Save best Stage 1 model
        if val_loss < best_stage1_loss:
            best_stage1_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': stage1_model.state_dict(),
                'optimizer_state_dict': trainer.stage1_optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'similarity': val_sim
            }, config.stage1_model_path)
            logger.info(f"Saved best Stage 1 model with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Update learning rate
        old_lr = trainer.stage1_optimizer.param_groups[0]['lr']
        trainer.stage1_scheduler.step(val_loss)
        new_lr = trainer.stage1_optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            logger.info(f"Stage 1 LR reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"Stage 1 early stopping after {config.patience} epochs without improvement")
            break
    
    # Load best Stage 1 model for Stage 2
    logger.info("Loading best Stage 1 model for Stage 2 training")
    checkpoint = torch.load(config.stage1_model_path, map_location=device)
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    stage1_model.eval()
    
    # STAGE 2: Contrastive Learning
    logger.info("="*50)
    logger.info("STAGE 2: Contrastive Learning Enhancement")
    logger.info("="*50)
    
    # Load or create Stage 2 dataset
    if os.path.exists(config.stage2_embeddings):
        logger.info(f"Loading Stage 2 embeddings from {config.stage2_embeddings}")
        stage2_dataset = UnifiedDataset(embeddings_file=config.stage2_embeddings, stage=2)
    else:
        logger.info(f"Creating Stage 2 dataset from {config.stage2_data}")
        stage2_dataset = UnifiedDataset(
            stage2_data_file=config.stage2_data,
            flag_model=flag_model,
            stage1_model=stage1_model,
            device=device,
            max_tables=config.max_tables,
            max_columns=config.max_columns,
            max_similar=config.max_similar,
            stage=2
        )
        stage2_dataset.save_embeddings(config.stage2_embeddings)
    
    # Split Stage 2 dataset
    train_size = int(0.9 * len(stage2_dataset))
    val_size = len(stage2_dataset) - train_size
    stage2_train, stage2_val = torch.utils.data.random_split(
        stage2_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    stage2_train_loader = DataLoader(
        stage2_train, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn_stage2,
        num_workers=0
    )
    stage2_val_loader = DataLoader(
        stage2_val, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn_stage2,
        num_workers=0
    )
    
    # Train Stage 2
    best_stage2_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.stage2_epochs):
        logger.info(f"\nStage 2 - Epoch {epoch + 1}/{config.stage2_epochs}")
        
        train_loss, train_cont, train_sim = trainer.train_stage2_epoch(stage2_train_loader)
        val_loss, val_cont, val_sim = trainer.validate_stage2(stage2_val_loader)
        
        stage2_metrics['train_losses'].append(train_loss)
        stage2_metrics['val_losses'].append(val_loss)
        stage2_metrics['train_contrastive'].append(train_cont)
        stage2_metrics['val_contrastive'].append(val_cont)
        stage2_metrics['train_similarity'].append(train_sim)
        stage2_metrics['val_similarity'].append(val_sim)
        
        logger.info(f"Train Loss: {train_loss:.4f} (Contrastive: {train_cont:.4f}, Similarity: {train_sim:.4f})")
        logger.info(f"Val Loss: {val_loss:.4f} (Contrastive: {val_cont:.4f}, Similarity: {val_sim:.4f})")
        logger.info(f"LR: {trainer.stage2_scheduler.get_last_lr()[0]:.6f}")
        
        # Save best Stage 2 model
        if val_loss < best_stage2_loss:
            best_stage2_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': stage2_model.state_dict(),
                'optimizer_state_dict': trainer.stage2_optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'contrastive_loss': val_cont,
                'similarity_loss': val_sim
            }, config.stage2_model_path)
            logger.info(f"Saved best Stage 2 model with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Stage 2 early stopping after {config.patience} epochs without improvement")
                break
        
        trainer.stage2_scheduler.step()
    
    # Save final models
    final_stage1_path = config.stage1_model_path.replace('.pth', '_final.pth')
    final_stage2_path = config.stage2_model_path.replace('.pth', '_final.pth')
    
    torch.save({
        'model_state_dict': stage1_model.state_dict(),
        'optimizer_state_dict': trainer.stage1_optimizer.state_dict(),
        'epoch': len(stage1_metrics['train_losses']) - 1,
        'loss': stage1_metrics['val_losses'][-1] if stage1_metrics['val_losses'] else float('inf'),
        'similarity': stage1_metrics['val_similarities'][-1] if stage1_metrics['val_similarities'] else 0.0
    }, final_stage1_path)
    
    torch.save({
        'model_state_dict': stage2_model.state_dict(),
        'optimizer_state_dict': trainer.stage2_optimizer.state_dict(),
        'epoch': len(stage2_metrics['train_losses']) - 1,
        'loss': stage2_metrics['val_losses'][-1] if stage2_metrics['val_losses'] else float('inf'),
        'contrastive_loss': stage2_metrics['val_contrastive'][-1] if stage2_metrics['val_contrastive'] else float('inf'),
        'similarity_loss': stage2_metrics['val_similarity'][-1] if stage2_metrics['val_similarity'] else float('inf')
    }, final_stage2_path)
    
    # Plot training curves
    plot_training_curves(stage1_metrics, stage2_metrics, config.output_dir)
    
    logger.info("="*50)
    logger.info("TRAINING COMPLETED")
    logger.info("="*50)
    logger.info(f"Best Stage 1 model saved at: {config.stage1_model_path}")
    logger.info(f"Best Stage 2 model saved at: {config.stage2_model_path}")
    logger.info(f"Final Stage 1 model saved at: {final_stage1_path}")
    logger.info(f"Final Stage 2 model saved at: {final_stage2_path}")
    logger.info(f"Training plots saved in: {config.output_dir}")
    
    return {
        'stage1_metrics': stage1_metrics,
        'stage2_metrics': stage2_metrics,
        'best_stage1_loss': best_stage1_loss,
        'best_stage2_loss': best_stage2_loss
    }


class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self):
        # Data paths
        self.stage1_data = './mini_SA_200.json'
        self.stage2_data = './mini_CL_200.json'
        self.stage1_embeddings = './embeddings/stage1_embeddings.pt'
        self.stage2_embeddings = './embeddings/stage2_embeddings.pt'
        self.flag_model_path = './plm/embeddingmodl'
        
        # Model save paths
        self.stage1_model_path = './SAR/models/best_schema_aware_model.pth'
        self.stage2_model_path = './SAR/models/best_contrastive_model.pth'
        
        # Output directories
        self.output_dir = './training_plots'
        self.log_dir = './logs'
        
        # Model parameters
        self.embed_dim = 1024
        self.num_heads = 8
        self.dropout = 0.1
        self.stage2_layers = 2
        
        # Data parameters
        self.max_tables = 10
        self.max_columns = 20
        self.max_similar = 3
        
        # Training parameters
        self.stage1_epochs = 20
        self.stage2_epochs = 15
        self.batch_size = 8
        self.stage1_lr = 5e-5
        self.stage2_lr = 5e-5
        self.temperature = 0.1
        self.similarity_weight = 0.3
        self.patience = 5
        
        # System parameters
        self.device = 'cuda'
        self.seed = 42
    
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
    
    def __str__(self):
        """String representation of config"""
        config_str = "Training Configuration:\n"
        config_str += "=" * 30 + "\n"
        for key, value in vars(self).items():
            config_str += f"{key}: {value}\n"
        config_str += "=" * 30
        return config_str


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Unified Schema-Aware and Contrastive Learning Training')
    
    # Data arguments
    parser.add_argument('--stage1_data', type=str, help='Path to Stage 1 training data')
    parser.add_argument('--stage2_data', type=str, help='Path to Stage 2 training data')
    parser.add_argument('--stage1_embeddings', type=str, help='Path to Stage 1 embeddings file')
    parser.add_argument('--stage2_embeddings', type=str, help='Path to Stage 2 embeddings file')
    parser.add_argument('--flag_model_path', type=str, help='Path to Flag embedding model')
    
    # Model arguments
    parser.add_argument('--stage1_model_path', type=str, help='Path to save Stage 1 model')
    parser.add_argument('--stage2_model_path', type=str, help='Path to save Stage 2 model')
    parser.add_argument('--embed_dim', type=int, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--stage2_layers', type=int, help='Number of layers in Stage 2 transformer')
    
    # Training arguments
    parser.add_argument('--stage1_epochs', type=int, help='Number of Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, help='Number of Stage 2 epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--stage1_lr', type=float, help='Stage 1 learning rate')
    parser.add_argument('--stage2_lr', type=float, help='Stage 2 learning rate')
    parser.add_argument('--temperature', type=float, help='Temperature for contrastive loss')
    parser.add_argument('--similarity_weight', type=float, help='Weight for similarity loss')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--output_dir', type=str, help='Output directory for plots')
    parser.add_argument('--log_dir', type=str, help='Log directory')
    
    # Data parameters
    parser.add_argument('--max_tables', type=int, help='Maximum number of tables')
    parser.add_argument('--max_columns', type=int, help='Maximum number of columns per table')
    parser.add_argument('--max_similar', type=int, help='Maximum number of similar samples')
    
    args = parser.parse_args()
    
    # Initialize config and update from args
    config = TrainingConfig()
    config.update_from_args(args)
    
    # Create necessary directories
    os.makedirs(os.path.dirname(config.stage1_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.stage2_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.stage1_embeddings), exist_ok=True)
    os.makedirs(os.path.dirname(config.stage2_embeddings), exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    print(config)
    
    # Start training
    results = train_unified_pipeline(config)
    
    print("\nTraining Results:")
    print(f"Best Stage 1 Validation Loss: {results['best_stage1_loss']:.4f}")
    print(f"Best Stage 2 Validation Loss: {results['best_stage2_loss']:.4f}")
    
    return results


if __name__ == "__main__":
    main()