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


class SchemaAwareDataset(Dataset):
    """Dataset for Schema-Aware Representation Learning"""
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
    
    def _process_data(self, data_file):
        """Process raw data and generate embeddings"""
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
                    table_column_embeds.append(torch.zeros_like(question_embed))
                
                column_embed_list.append(torch.stack(table_column_embeds))
            
            while len(table_embed_list) < self.max_tables:
                table_embed_list.append(torch.zeros_like(question_embed))
                column_embed_list.append(torch.zeros(self.max_columns, question_embed.shape[0]))
            
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
    """Trainer for Schema-Aware Representation Learning"""
    def __init__(self, 
                 model: SchemaAwareModel,
                 device: torch.device,
                 learning_rate: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_similarity = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
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
            
            loss = self.criterion(schema_aware_embeds, sql_embeds)
            similarity = F.cosine_similarity(schema_aware_embeds, sql_embeds, dim=-1).mean()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_similarity += similarity.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_similarity = total_similarity / num_batches
        return avg_loss, avg_similarity
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_similarity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                question_embeds = batch['question_embed'].to(self.device)
                table_embeds = batch['table_embeds'].to(self.device)
                column_embeds = batch['column_embeds'].to(self.device)
                sql_embeds = batch['sql_embed'].to(self.device)
                table_masks = batch['table_masks'].to(self.device)
                column_masks = batch['column_masks'].to(self.device)
                
                schema_aware_embeds, _ = self.model(
                    question_embeds, table_embeds, column_embeds, table_masks, column_masks
                )
                
                loss = self.criterion(schema_aware_embeds, sql_embeds)
                similarity = F.cosine_similarity(schema_aware_embeds, sql_embeds, dim=-1).mean()
                
                total_loss += loss.item()
                total_similarity += similarity.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_similarity = total_similarity / num_batches
        return avg_loss, avg_similarity


def train_schema_aware_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, 
                           device='cuda', save_path='best_model.pth', 
                           output_dir='training_plots'):
    """Train the schema-aware model with plotting"""
    trainer = SchemaAwareTrainer(model, device, lr)
    
    train_losses = []
    val_losses = []
    train_similarities = []
    val_similarities = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_sim = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        train_similarities.append(train_sim)
        
        val_loss, val_sim = trainer.validate(val_loader)
        val_losses.append(val_loss)
        val_similarities.append(val_sim)
        
        print(f"Train Loss: {train_loss:.4f}, Train Similarity: {train_sim:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Similarity: {val_sim:.4f}")
        print(f"LR: {trainer.scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'similarity': val_sim
            }, save_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")
        
        trainer.scheduler.step()
    
    # Save final model
    final_save_path = save_path.replace('.pth', '_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'epoch': num_epochs - 1,
        'loss': val_losses[-1],
        'similarity': val_similarities[-1]
    }, final_save_path)
    
    # Create plots
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_similarities, label='Train Similarity')
    plt.plot(range(1, num_epochs + 1), val_similarities, label='Val Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title('Schema-Aware Embedding Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'similarity.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Schema-Aware Model Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()


def main():
    """Main training function"""
    # Configuration - use relative paths
    embeddings_file = './embeddings/schema_aware_embeddings.pt'
    data_file = './datasets/train.json'
    model_save_path = './SAR/models/best_schema_aware_model.pth'
    output_dir = './training_plots/schema_aware'
    flag_model_path = './plm/bge-large-en-v1.5'
    
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
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = SchemaAwareModel(
        embed_dim=1024,
        num_heads=8,
        dropout=0.1
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train_schema_aware_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        lr=1e-4,
        device=device,
        save_path=model_save_path,
        output_dir=output_dir
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()



# {
#   "question": "What is the name of the singer?",
#   "query": "SELECT name FROM singer",
#   "schema": {
#     "tables": ["singer", "album"],
#     "columns": {
#       "singer": ["id", "name", "age"],
#       "album": ["id", "title", "singer_id"]
#     }
#   }
# }