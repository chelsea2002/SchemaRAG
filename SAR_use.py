import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from FlagEmbedding import FlagModel
from typing import List, Dict, Tuple, Optional
import math
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class SchemaAwareModel(nn.Module):
    """Schema-Aware Representation Learning Model"""
    def __init__(self, 
                 embed_dim: int = 1024, 
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super(SchemaAwareModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention for table-column interaction
        self.table_column_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head attention for question-table interaction
        self.question_table_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.table_proj = nn.Linear(embed_dim, embed_dim)
        self.column_proj = nn.Linear(embed_dim, embed_dim)
        self.question_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
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
        
        # Generate column-aware table embeddings
        column_aware_table_embeds = []
        
        for i in range(max_tables):
            table_i = table_embeds[:, i:i+1, :]
            columns_i = column_embeds[:, i, :, :]
            col_mask_i = column_masks[:, i, :]
            
            table_i_proj = self.table_proj(table_i)
            columns_i_proj = self.column_proj(columns_i)
            
            attn_output, _ = self.table_column_attention(
                query=table_i_proj,
                key=columns_i_proj,
                value=columns_i_proj,
                key_padding_mask=~col_mask_i.bool()
            )
            
            column_aware_table_i = self.layer_norm1(table_i + attn_output)
            column_aware_table_embeds.append(column_aware_table_i)
        
        column_aware_tables = torch.cat(column_aware_table_embeds, dim=1)
        
        # Generate schema-aware embedding
        question_proj = self.question_proj(question_embed.unsqueeze(1))
        
        schema_aware_output, attention_weights = self.question_table_attention(
            query=question_proj,
            key=column_aware_tables,
            value=column_aware_tables,
            key_padding_mask=~table_masks.bool()
        )
        
        schema_aware_embed = self.layer_norm2(question_proj + schema_aware_output)
        schema_aware_embed = self.output_proj(schema_aware_embed.squeeze(1))
        
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
        
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
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


class SARRetrieval:
    """SAR Model Inference and Retrieval System"""
    
    def __init__(self,
                 stage1_model_path: str,
                 stage2_model_path: str,
                 flag_model_path: str,
                 supervised_data_path: str,
                 max_tables: int = 10,
                 max_columns: int = 20,
                 device: str = 'cuda'):
        """
        Initialize SAR Retrieval System
        
        Args:
            stage1_model_path: Path to trained SchemaAwareModel
            stage2_model_path: Path to trained ContrastiveLearningModel
            flag_model_path: Path to FlagModel for encoding
            supervised_data_path: Path to supervised dataset for retrieval
            max_tables: Maximum number of tables to consider
            max_columns: Maximum number of columns per table
            device: Device to run models on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_tables = max_tables
        self.max_columns = max_columns
        
        print(f"Using device: {self.device}")
        
        # Load FlagModel for encoding
        print("Loading FlagModel...")
        self.flag_model = FlagModel(flag_model_path, use_fp16=True)
        
        # Load Stage 1 Model (Schema-Aware)
        print("Loading Stage 1 Model...")
        self.stage1_model = SchemaAwareModel(
            embed_dim=1024,
            num_heads=8,
            dropout=0.1
        ).to(self.device)
        
        stage1_checkpoint = torch.load(stage1_model_path, map_location=self.device)
        self.stage1_model.load_state_dict(stage1_checkpoint['model_state_dict'])
        self.stage1_model.eval()
        
        # Load Stage 2 Model (Contrastive Learning)
        print("Loading Stage 2 Model...")
        self.stage2_model = ContrastiveLearningModel(
            embed_dim=1024,
            num_layers=3,
            num_heads=8,
            dropout=0.1
        ).to(self.device)
        
        stage2_checkpoint = torch.load(stage2_model_path, map_location=self.device)
        self.stage2_model.load_state_dict(stage2_checkpoint['model_state_dict'])
        self.stage2_model.eval()
        
        # Load supervised dataset
        print("Loading supervised dataset...")
        self.supervised_data = self._load_supervised_data(supervised_data_path)
        
        # Precompute embeddings for supervised data
        print("Precomputing embeddings for supervised data...")
        self._precompute_supervised_embeddings()
        
        print("SAR Retrieval System initialized successfully!")
    
    def _load_supervised_data(self, data_path: str) -> List[Dict]:
        """Load supervised dataset"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _process_schema(self, schema: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process schema information into embeddings and masks
        
        Args:
            schema: Schema dictionary with 'tables' and 'columns'
        
        Returns:
            table_embeds: [1, max_tables, embed_dim]
            column_embeds: [1, max_tables, max_columns, embed_dim]
            table_masks: [1, max_tables]
            column_masks: [1, max_tables, max_columns]
        """
        tables = schema['tables']
        table_columns = schema['columns']
        
        table_embed_list = []
        column_embed_list = []
        table_mask = torch.zeros(self.max_tables, dtype=torch.bool)
        column_mask = torch.zeros(self.max_tables, self.max_columns, dtype=torch.bool)
        
        for i, table in enumerate(tables[:self.max_tables]):
            # Encode table
            table_embed = torch.tensor(
                self.flag_model.encode(f"Table: {table}"), 
                dtype=torch.float32, 
                device=self.device
            )
            table_embed_list.append(table_embed)
            table_mask[i] = True
            
            # Encode columns for this table
            columns = table_columns.get(table, [])
            table_column_embeds = []
            
            for j, column in enumerate(columns[:self.max_columns]):
                column_embed = torch.tensor(
                    self.flag_model.encode(f"Column: {column} in {table}"), 
                    dtype=torch.float32,
                    device=self.device
                )
                table_column_embeds.append(column_embed)
                column_mask[i, j] = True
            
            # Pad columns if necessary
            while len(table_column_embeds) < self.max_columns:
                table_column_embeds.append(torch.zeros_like(table_embed))
            
            column_embed_list.append(torch.stack(table_column_embeds))
        
        # Pad tables if necessary
        while len(table_embed_list) < self.max_tables:
            embed_dim = table_embed_list[0].shape[0] if table_embed_list else 1024
            table_embed_list.append(torch.zeros(embed_dim, device=self.device))
            column_embed_list.append(torch.zeros(self.max_columns, embed_dim, device=self.device))
        
        table_embeds = torch.stack(table_embed_list).unsqueeze(0)  # [1, max_tables, embed_dim]
        column_embeds = torch.stack(column_embed_list).unsqueeze(0)  # [1, max_tables, max_columns, embed_dim]
        table_masks = table_mask.unsqueeze(0).to(self.device)  # [1, max_tables]
        column_masks = column_mask.unsqueeze(0).to(self.device)  # [1, max_tables, max_columns]
        
        return table_embeds, column_embeds, table_masks, column_masks
    
    def _precompute_supervised_embeddings(self):
        """Precompute embeddings for all supervised data"""
        self.supervised_embeddings = []
        self.supervised_questions = []
        self.supervised_sqls = []
        
        with torch.no_grad():
            for item in tqdm(self.supervised_data, desc="Precomputing supervised embeddings"):
                question = item['question']
                sql = item['query']
                schema = item.get('schema', {})
                
                # Get final enhanced embedding using SAR
                enhanced_embed = self.get_enhanced_embedding(question, schema)
                
                self.supervised_embeddings.append(enhanced_embed.cpu().numpy())
                self.supervised_questions.append(question)
                self.supervised_sqls.append(sql)
        
        self.supervised_embeddings = np.array(self.supervised_embeddings)
        print(f"Precomputed embeddings for {len(self.supervised_embeddings)} supervised samples")
    
    def get_enhanced_embedding(self, question: str, schema: Dict) -> torch.Tensor:
        """
        Get enhanced embedding using both stage models
        
        Args:
            question: Input question string
            schema: Schema dictionary
        
        Returns:
            enhanced_embed: Enhanced embedding from Stage 2 model
        """
        with torch.no_grad():
            # Encode question
            question_embed = torch.tensor(
                self.flag_model.encode(question), 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)  # [1, embed_dim]
            
            # Process schema
            table_embeds, column_embeds, table_masks, column_masks = self._process_schema(schema)
            
            # Stage 1: Get schema-aware embedding
            schema_aware_embed, attention_weights = self.stage1_model(
                question_embed, table_embeds, column_embeds, table_masks, column_masks
            )
            
            # Stage 2: Get enhanced embedding
            enhanced_embed = self.stage2_model(question_embed, schema_aware_embed)
            
            return enhanced_embed.squeeze(0)  # [embed_dim]
    
    def retrieve_similar_samples(self, 
                                question: str, 
                                schema: Dict, 
                                n: int = 5) -> List[Dict]:
        """
        Retrieve n most similar samples from supervised dataset
        
        Args:
            question: Input question
            schema: Schema dictionary
            n: Number of similar samples to retrieve
        
        Returns:
            List of similar samples with similarity scores
        """
        # Get enhanced embedding for input question
        query_embed = self.get_enhanced_embedding(question, schema)
        query_embed_np = query_embed.cpu().numpy().reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embed_np, self.supervised_embeddings)[0]
        
        # Get top n similar samples
        top_indices = np.argsort(similarities)[::-1][:n]
        
        results = []
        for idx in top_indices:
            results.append({
                'question': self.supervised_questions[idx],
                'sql': self.supervised_sqls[idx],
                'similarity': float(similarities[idx]),
                'index': int(idx)
            })
        
        return results
    
    def batch_retrieve(self, 
                      queries: List[Tuple[str, Dict]], 
                      n: int = 5) -> List[List[Dict]]:
        """
        Batch retrieval for multiple queries
        
        Args:
            queries: List of (question, schema) tuples
            n: Number of similar samples to retrieve for each query
        
        Returns:
            List of retrieval results for each query
        """
        results = []
        for question, schema in tqdm(queries, desc="Batch retrieval"):
            similar_samples = self.retrieve_similar_samples(question, schema, n)
            results.append(similar_samples)
        
        return results


def main():
    """Example usage of SAR Retrieval System"""
    
    # Configuration paths
    stage1_model_path = './SAR/models/best_schema_aware_model.pth'
    stage2_model_path = './SAR/models/best_contrastive_model.pth'
    flag_model_path = './plm/bge-large-en-v1.5'
    supervised_data_path = './datasets/RAG_data.json'
    
    # Initialize SAR Retrieval System
    sar_retrieval = SARRetrieval(
        stage1_model_path=stage1_model_path,
        stage2_model_path=stage2_model_path,
        flag_model_path=flag_model_path,
        supervised_data_path=supervised_data_path,
        device='cuda'
    )
    
    # Example query
    # question = "What is the name of the singer?"
    # schema = {
    #     "tables": ["singer", "album"],
    #     "columns": {
    #         "singer": ["id", "name", "age"],
    #         "album": ["id", "title", "singer_id"]
    #     }
    # }

    
    # similar_samples = sar_retrieval.retrieve_similar_samples(question, schema, n=5)
    
    


if __name__ == "__main__":
    main()