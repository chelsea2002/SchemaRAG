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
import argparse
import hashlib
import pickle


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


class SARRetrieval:
    """SAR Model Inference and Retrieval System with JSON I/O and Embedding Cache"""
    
    def __init__(self,
                 stage1_model_path: str,
                 stage2_model_path: str,
                 flag_model_path: str,
                 supervised_data_path: str,
                 max_tables: int = 10,
                 max_columns: int = 20,
                 device: str = 'cuda',
                 cache_dir: str = './embedding_cache'):
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
            cache_dir: Directory to store embedding cache files
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_tables = max_tables
        self.max_columns = max_columns
        self.cache_dir = cache_dir
        self.supervised_data_path = supervised_data_path
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
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
            num_layers=2,
            num_heads=8,
            dropout=0.1
        ).to(self.device)
        
        stage2_checkpoint = torch.load(stage2_model_path, map_location=self.device)
        self.stage2_model.load_state_dict(stage2_checkpoint['model_state_dict'])
        self.stage2_model.eval()
        
        # Load supervised dataset
        print("Loading supervised dataset...")
        self.supervised_data = self._load_supervised_data(supervised_data_path)
        
        # Load or precompute embeddings for supervised data
        print("Loading/Computing embeddings for supervised data...")
        self._load_or_compute_supervised_embeddings()
        
        print("SAR Retrieval System initialized successfully!")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for a file based on its path and modification time"""
        abs_path = os.path.abspath(file_path)
        filename = os.path.basename(abs_path)
        
        # Create hash based on filename and file modification time (if file exists)
        hash_components = [filename]
        if os.path.exists(abs_path):
            hash_components.append(str(os.path.getmtime(abs_path)))
        
        hash_string = '_'.join(hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _get_cache_filename(self, data_file_path: str) -> str:
        """Generate cache filename based on data file"""
        file_hash = self._get_file_hash(data_file_path)
        base_name = os.path.splitext(os.path.basename(data_file_path))[0]
        return f"{base_name}_{file_hash}_embeddings.pkl"
    
    def _save_embeddings_to_cache(self, cache_path: str, embeddings_data: Dict):
        """Save embeddings data to cache file"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings_data, f)
            print(f"Embeddings saved to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save embeddings to cache: {e}")
    
    def _load_embeddings_from_cache(self, cache_path: str) -> Optional[Dict]:
        """Load embeddings data from cache file"""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    embeddings_data = pickle.load(f)
                print(f"Embeddings loaded from cache: {cache_path}")
                return embeddings_data
        except Exception as e:
            print(f"Warning: Failed to load embeddings from cache: {e}")
        return None
    
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
    
    def _load_or_compute_supervised_embeddings(self):
        """Load embeddings from cache or compute if not available"""
        cache_filename = self._get_cache_filename(self.supervised_data_path)
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # Try to load from cache first
        cached_data = self._load_embeddings_from_cache(cache_path)
        
        if cached_data is not None:
            # Load from cache
            self.supervised_embeddings = cached_data['embeddings']
            self.supervised_questions = cached_data['questions']
            self.supervised_sqls = cached_data['sqls']
            print(f"Loaded embeddings for {len(self.supervised_embeddings)} supervised samples from cache")
        else:
            # Compute embeddings
            print("Cache not found. Computing embeddings...")
            self._compute_supervised_embeddings()
            
            # Save to cache
            embeddings_data = {
                'embeddings': self.supervised_embeddings,
                'questions': self.supervised_questions,
                'sqls': self.supervised_sqls,
                'metadata': {
                    'data_file': self.supervised_data_path,
                    'total_samples': len(self.supervised_embeddings),
                    'embed_dim': self.supervised_embeddings.shape[1] if len(self.supervised_embeddings) > 0 else 0
                }
            }
            self._save_embeddings_to_cache(cache_path, embeddings_data)
    
    def _compute_supervised_embeddings(self):
        """Compute embeddings for all supervised data"""
        self.supervised_embeddings = []
        self.supervised_questions = []
        self.supervised_sqls = []
        
        with torch.no_grad():
            for item in tqdm(self.supervised_data, desc="Computing supervised embeddings"):
                question = item['question']
                sql = item.get('query', item.get('sql', ''))  # Handle different key names
                schema = item.get('schema', {})
                
                # Get final enhanced embedding using SAR
                enhanced_embed = self.get_enhanced_embedding(question, schema)
                
                self.supervised_embeddings.append(enhanced_embed.cpu().numpy())
                self.supervised_questions.append(question)
                self.supervised_sqls.append(sql)
        
        self.supervised_embeddings = np.array(self.supervised_embeddings)
        print(f"Computed embeddings for {len(self.supervised_embeddings)} supervised samples")
    
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
                'rag_text': self.supervised_questions[idx],
                'rag_sql': self.supervised_sqls[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def process_test_file(self, 
                         test_file_path: str, 
                         output_file_path: str, 
                         k: int = 5):
        """
        Process test file and generate retrieval results
        
        Args:
            test_file_path: Path to test JSON file
            output_file_path: Path to output JSON file
            k: Number of top matches to retrieve
        """
        # Load test data
        print(f"Loading test data from {test_file_path}")
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = []
        
        print(f"Processing {len(test_data)} test cases...")
        for item in tqdm(test_data, desc="Processing test cases"):
            test_question = item['question']
            test_sql = item.get('query', item.get('sql', ''))  # Handle different key names
            test_schema = item.get('schema', {})
            
            # Retrieve similar samples
            top_k_matches = self.retrieve_similar_samples(
                test_question, test_schema, n=k
            )
            
            # Format result
            result = {
                'test_text': test_question,
                'test_sql': test_sql,
                'top_k_matches': top_k_matches
            }
            
            results.append(result)
        
        # Save results
        print(f"Saving results to {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully processed {len(results)} test cases")
        return results
    
    def clear_cache(self, data_file_path: Optional[str] = None):
        """
        Clear embedding cache
        
        Args:
            data_file_path: Specific file to clear cache for. If None, clear all cache.
        """
        if data_file_path:
            # Clear cache for specific file
            cache_filename = self._get_cache_filename(data_file_path)
            cache_path = os.path.join(self.cache_dir, cache_filename)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"Cache cleared for {data_file_path}")
            else:
                print(f"No cache found for {data_file_path}")
        else:
            # Clear all cache
            if os.path.exists(self.cache_dir):
                import shutil
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                print("All cache cleared")
    
    def list_cache_files(self):
        """List all cache files"""
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('_embeddings.pkl')]
            print(f"Found {len(cache_files)} cache files:")
            for cache_file in cache_files:
                cache_path = os.path.join(self.cache_dir, cache_file)
                file_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
                print(f"  - {cache_file} ({file_size:.2f} MB)")
        else:
            print("Cache directory does not exist")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="SAR Retrieval System with JSON I/O and Embedding Cache")
    
    parser.add_argument('--stage1_model', type=str, required=True,
                       help='Path to Stage 1 model checkpoint')
    parser.add_argument('--stage2_model', type=str, required=True,
                       help='Path to Stage 2 model checkpoint')
    parser.add_argument('--flag_model', type=str, required=True,
                       help='Path to FlagModel')
    parser.add_argument('--supervised_data', type=str, required=True,
                       help='Path to supervised dataset JSON file')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test dataset JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output results JSON file')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of top matches to retrieve (default: 5)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--max_tables', type=int, default=10,
                       help='Maximum number of tables (default: 10)')
    parser.add_argument('--max_columns', type=int, default=20,
                       help='Maximum number of columns per table (default: 20)')
    parser.add_argument('--cache_dir', type=str, default='./embedding_cache',
                       help='Directory to store embedding cache files (default: ./embedding_cache)')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear all embedding cache before processing')
    parser.add_argument('--list_cache', action='store_true',
                       help='List all cache files and exit')
    
    args = parser.parse_args()
    
    # Initialize SAR Retrieval System
    print("Initializing SAR Retrieval System...")
    sar_retrieval = SARRetrieval(
        stage1_model_path=args.stage1_model,
        stage2_model_path=args.stage2_model,
        flag_model_path=args.flag_model,
        supervised_data_path=args.supervised_data,
        max_tables=args.max_tables,
        max_columns=args.max_columns,
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    # Handle cache operations
    if args.list_cache:
        sar_retrieval.list_cache_files()
        return
    
    if args.clear_cache:
        sar_retrieval.clear_cache()
        print("Cache cleared. Reinitializing embeddings...")
        # Reinitialize embeddings after clearing cache
        sar_retrieval._load_or_compute_supervised_embeddings()
    
    # Process test file
    results = sar_retrieval.process_test_file(
        test_file_path=args.test_file,
        output_file_path=args.output_file,
        k=args.k
    )
    
    # Print sample results
    if results:
        print("\nSample result:")
        sample = results[0]
        print(f"Test question: {sample['test_text']}")
        print(f"Test SQL: {sample['test_sql']}")
        print("Top matches:")
        for i, match in enumerate(sample['top_k_matches'][:3]):
            print(f"  {i+1}. {match['rag_text']} (similarity: {match['similarity']:.4f})")


def example_usage():
    """Example usage function"""
    # Configuration paths
    stage1_model_path = './SAR/models/best_schema_aware_model.pth'
    stage2_model_path = './SAR/models/best_contrastive_model.pth'
    flag_model_path = './plm/embeddingmodels'
    supervised_data_path = './RAG_Spider.json'
    test_file_path = './dev_spider.json'
    output_file_path = './retrieval_results.json'
    cache_dir = './embedding_cache'
    
    # Initialize SAR Retrieval System with cache
    sar_retrieval = SARRetrieval(
        stage1_model_path=stage1_model_path,
        stage2_model_path=stage2_model_path,
        flag_model_path=flag_model_path,
        supervised_data_path=supervised_data_path,
        device='cuda',
        cache_dir=cache_dir
    )
    
    # List current cache files
    print("\nCurrent cache files:")
    sar_retrieval.list_cache_files()
    
    # Process test file (will use cached embeddings if available)
    results = sar_retrieval.process_test_file(
        test_file_path=test_file_path,
        output_file_path=output_file_path,
        k=3
    )
    
    print(f"Processed {len(results)} test cases")
    
    # Example: Clear cache for specific file
    # sar_retrieval.clear_cache(supervised_data_path)
    
    # Example: Clear all cache
    # sar_retrieval.clear_cache()


if __name__ == "__main__":
    # Uncomment the line below to run the example instead of main
    # example_usage()
    # main()
