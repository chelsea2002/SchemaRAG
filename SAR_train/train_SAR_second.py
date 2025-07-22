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


class ContrastiveDataset(Dataset):
    """Dataset for Stage 2 Contrastive Learning with predefined similar samples"""
    def __init__(self, 
                 embeddings_file: str = None,
                 data_file: str = None,
                 flag_model: FlagModel = None,
                 stage1_embeddings_file: str = None):
        
        self.schema_aware_embeds = None
        if stage1_embeddings_file and os.path.exists(stage1_embeddings_file):
            print(f"Loading Stage 1 embeddings from {stage1_embeddings_file}...")
            stage1_data = torch.load(stage1_embeddings_file)
            self.question_to_schema_embed = {}
            for i, question in enumerate(stage1_data['questions']):
                self.question_to_schema_embed[question] = stage1_data['schema_aware_embeds'][i]
        
        if embeddings_file and os.path.exists(embeddings_file):
            print(f"Loading precomputed embeddings from {embeddings_file}...")
            embeddings = torch.load(embeddings_file)
            self.questions = embeddings['questions']
            self.sqls = embeddings['sqls']
            self.question_embeds = embeddings['question_embeds']
            self.sql_embeds = embeddings['sql_embeds']
            self.similar_questions = embeddings['similar_questions']
            self.similar_sqls = embeddings['similar_sqls']
            self.similar_question_embeds = embeddings['similar_question_embeds']
            self.similar_sql_embeds = embeddings['similar_sql_embeds']
            
            if hasattr(self, 'question_to_schema_embed'):
                self.schema_aware_embeds = []
                self.similar_schema_aware_embeds = []
                
                for i, question in enumerate(self.questions):
                    if question in self.question_to_schema_embed:
                        self.schema_aware_embeds.append(self.question_to_schema_embed[question])
                    else:
                        self.schema_aware_embeds.append(self.question_embeds[i])
                    
                    similar_schema_embeds = []
                    for sim_question in self.similar_questions[i]:
                        if sim_question in self.question_to_schema_embed:
                            similar_schema_embeds.append(self.question_to_schema_embed[sim_question])
                        else:
                            similar_schema_embeds.append(self.schema_aware_embeds[-1])
                    
                    while len(similar_schema_embeds) < 3:
                        similar_schema_embeds.append(self.schema_aware_embeds[-1])
                    
                    self.similar_schema_aware_embeds.append(torch.stack(similar_schema_embeds[:3]))
            else:
                self.schema_aware_embeds = self.question_embeds
                self.similar_schema_aware_embeds = self.similar_question_embeds
                
        elif data_file and flag_model:
            print(f"Processing data from {data_file}...")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.flag_model = flag_model
            self.questions = []
            self.sqls = []
            self.question_embeds = []
            self.sql_embeds = []
            self.similar_questions = []
            self.similar_sqls = []
            self.similar_question_embeds = []
            self.similar_sql_embeds = []
            self.schema_aware_embeds = []
            self.similar_schema_aware_embeds = []
            
            print("Processing dataset...")
            for item in tqdm(data, desc="Encoding texts and SQLs"):
                main_question = item['question']
                main_sql = item['query']
                main_question_embed = self.flag_model.encode(main_question)
                main_sql_embed = self.flag_model.encode(main_sql)
                
                self.questions.append(main_question)
                self.sqls.append(main_sql)
                self.question_embeds.append(torch.tensor(main_question_embed, dtype=torch.float32))
                self.sql_embeds.append(torch.tensor(main_sql_embed, dtype=torch.float32))
                
                if hasattr(self, 'question_to_schema_embed') and main_question in self.question_to_schema_embed:
                    schema_embed = self.question_to_schema_embed[main_question]
                else:
                    schema_embed = torch.tensor(main_question_embed, dtype=torch.float32)
                self.schema_aware_embeds.append(schema_embed)
                
                similar_questions = []
                similar_sqls = []
                similar_question_embeds = []
                similar_sql_embeds = []
                similar_schema_embeds = []
                
                for sim_item in item['similar']:
                    sim_question = sim_item['question']
                    sim_sql = sim_item['query']
                    sim_question_embed = self.flag_model.encode(sim_question)
                    sim_sql_embed = self.flag_model.encode(sim_sql)
                    
                    similar_questions.append(sim_question)
                    similar_sqls.append(sim_sql)
                    similar_question_embeds.append(torch.tensor(sim_question_embed, dtype=torch.float32))
                    similar_sql_embeds.append(torch.tensor(sim_sql_embed, dtype=torch.float32))
                    
                    if hasattr(self, 'question_to_schema_embed') and sim_question in self.question_to_schema_embed:
                        sim_schema_embed = self.question_to_schema_embed[sim_question]
                    else:
                        sim_schema_embed = torch.tensor(sim_question_embed, dtype=torch.float32)
                    similar_schema_embeds.append(sim_schema_embed)
                
                while len(similar_questions) < 3:
                    similar_questions.append(similar_questions[0] if similar_questions else main_question)
                    similar_sqls.append(similar_sqls[0] if similar_sqls else main_sql)
                    similar_question_embeds.append(similar_question_embeds[0] if similar_question_embeds else self.question_embeds[-1])
                    similar_sql_embeds.append(similar_sql_embeds[0] if similar_sql_embeds else self.sql_embeds[-1])
                    similar_schema_embeds.append(similar_schema_embeds[0] if similar_schema_embeds else schema_embed)
                
                self.similar_questions.append(similar_questions[:3])
                self.similar_sqls.append(similar_sqls[:3])
                self.similar_question_embeds.append(torch.stack(similar_question_embeds[:3]))
                self.similar_sql_embeds.append(torch.stack(similar_sql_embeds[:3]))
                self.similar_schema_aware_embeds.append(torch.stack(similar_schema_embeds[:3]))
        else:
            raise ValueError("Must provide either embeddings_file or both data_file and flag_model")
    
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
            'similar_questions': self.similar_questions[idx],
            'similar_sqls': self.similar_sqls[idx]
        }
    
    def save_embeddings(self, save_path):
        """Save embeddings to file"""
        embeddings = {
            'questions': self.questions,
            'sqls': self.sqls,
            'question_embeds': self.question_embeds,
            'sql_embeds': self.sql_embeds,
            'schema_aware_embeds': self.schema_aware_embeds,
            'similar_questions': self.similar_questions,
            'similar_sqls': self.similar_sqls,
            'similar_question_embeds': self.similar_question_embeds,
            'similar_sql_embeds': self.similar_sql_embeds,
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
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )
        
    def contrastive_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using InfoNCE"""
        batch_size = embeddings.shape[0]
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = torch.arange(batch_size).to(self.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
    
    def similarity_loss(self, 
                       main_embeddings: torch.Tensor,
                       similar_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity loss to bring similar samples closer"""
        batch_size, num_similar, embed_dim = similar_embeddings.shape
        
        main_embeddings = F.normalize(main_embeddings, dim=1)
        similar_embeddings = F.normalize(similar_embeddings, dim=2)
        
        total_loss = 0.0
        
        for i in range(num_similar):
            similarities = F.cosine_similarity(
                main_embeddings, 
                similar_embeddings[:, i, :], 
                dim=1
            )
            loss = 1.0 - similarities.mean()
            total_loss += loss
        
        return total_loss / num_similar
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_similarity_loss = 0.0
        total_question_sim = 0.0
        total_schema_sim = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training Stage 2"):
            question_embeds = batch['question_embed'].to(self.device)
            schema_aware_embeds = batch['schema_aware_embed'].to(self.device)
            similar_question_embeds = batch['similar_question_embeds'].to(self.device)
            similar_schema_aware_embeds = batch['similar_schema_aware_embeds'].to(self.device)
            
            self.optimizer.zero_grad()
            
            enhanced_embeds = self.model(question_embeds, schema_aware_embeds)
            
            question_sim = F.cosine_similarity(enhanced_embeds, question_embeds, dim=1).mean().item()
            schema_sim = F.cosine_similarity(enhanced_embeds, schema_aware_embeds, dim=1).mean().item()
            total_question_sim += question_sim
            total_schema_sim += schema_sim
            
            batch_size, max_similar, embed_dim = similar_question_embeds.shape
            similar_question_flat = similar_question_embeds.view(-1, embed_dim)
            similar_schema_flat = similar_schema_aware_embeds.view(-1, embed_dim)
            
            enhanced_similar_flat = self.model(similar_question_flat, similar_schema_flat)
            enhanced_similar = enhanced_similar_flat.view(batch_size, max_similar, embed_dim)
            
            contrastive_loss = self.contrastive_loss(enhanced_embeds)
            similarity_loss = self.similarity_loss(enhanced_embeds, enhanced_similar)
            
            total_batch_loss = contrastive_loss + self.similarity_weight * similarity_loss
            
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_similarity_loss += similarity_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_similarity_loss = total_similarity_loss / num_batches
        avg_question_sim = total_question_sim / num_batches
        avg_schema_sim = total_schema_sim / num_batches
        
        return avg_loss, avg_contrastive_loss, avg_similarity_loss, avg_question_sim, avg_schema_sim
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, float, float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_similarity_loss = 0.0
        total_question_sim = 0.0
        total_schema_sim = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating Stage 2"):
                question_embeds = batch['question_embed'].to(self.device)
                schema_aware_embeds = batch['schema_aware_embed'].to(self.device)
                similar_question_embeds = batch['similar_question_embeds'].to(self.device)
                similar_schema_aware_embeds = batch['similar_schema_aware_embeds'].to(self.device)
                
                enhanced_embeds = self.model(question_embeds, schema_aware_embeds)
                
                question_sim = F.cosine_similarity(enhanced_embeds, question_embeds, dim=1).mean().item()
                schema_sim = F.cosine_similarity(enhanced_embeds, schema_aware_embeds, dim=1).mean().item()
                total_question_sim += question_sim
                total_schema_sim += schema_sim
                
                batch_size, max_similar, embed_dim = similar_question_embeds.shape
                similar_question_flat = similar_question_embeds.view(-1, embed_dim)
                similar_schema_flat = similar_schema_aware_embeds.view(-1, embed_dim)
                
                enhanced_similar_flat = self.model(similar_question_flat, similar_schema_flat)
                enhanced_similar = enhanced_similar_flat.view(batch_size, max_similar, embed_dim)
                
                contrastive_loss = self.contrastive_loss(enhanced_embeds)
                similarity_loss = self.similarity_loss(enhanced_embeds, enhanced_similar)
                
                total_batch_loss = contrastive_loss + self.similarity_weight * similarity_loss
                
                total_loss += total_batch_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_similarity_loss += similarity_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_similarity_loss = total_similarity_loss / num_batches
        avg_question_sim = total_question_sim / num_batches
        avg_schema_sim = total_schema_sim / num_batches
        
        return avg_loss, avg_contrastive_loss, avg_similarity_loss, avg_question_sim, avg_schema_sim


def train_stage2_contrastive_model(data_file: str = None,
                                 embeddings_file: str = None,
                                 stage1_embeddings_file: str = None,
                                 flag_model_path: str = None,
                                 num_epochs: int = 15,
                                 batch_size: int = 32,
                                 learning_rate: float = 1e-4,
                                 temperature: float = 0.05,
                                 device: str = 'cuda',
                                 save_path: str = './SAR/models/stage2_contrastive_model.pth',
                                 output_dir: str = './training_plots/stage2'):
    """Train Stage 2 Contrastive Learning Model"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if embeddings_file and os.path.exists(embeddings_file):
        dataset = ContrastiveDataset(
            embeddings_file=embeddings_file,
            stage1_embeddings_file=stage1_embeddings_file
        )
    elif data_file and flag_model_path:
        flag_model = FlagModel(flag_model_path, use_fp16=True)
        dataset = ContrastiveDataset(
            data_file=data_file,
            flag_model=flag_model,
            stage1_embeddings_file=stage1_embeddings_file
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ContrastiveLearningModel(
        embed_dim=1024,
        num_layers=3,
        num_heads=8,
        dropout=0.1
    )
    
    print(f"Stage 2 model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    trainer = ContrastiveTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        temperature=temperature,
        similarity_weight=0.5
    )
    
    train_losses = []
    val_losses = []
    train_contrastive_losses = []
    val_contrastive_losses = []
    train_similarity_losses = []
    val_similarity_losses = []
    train_question_sims = []
    val_question_sims = []
    train_schema_sims = []
    val_schema_sims = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_cont_loss, train_sim_loss, train_q_sim, train_s_sim = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        train_contrastive_losses.append(train_cont_loss)
        train_similarity_losses.append(train_sim_loss)
        train_question_sims.append(train_q_sim)
        train_schema_sims.append(train_s_sim)
        
        val_loss, val_cont_loss, val_sim_loss, val_q_sim, val_s_sim = trainer.validate(val_loader)
        val_losses.append(val_loss)
        val_contrastive_losses.append(val_cont_loss)
        val_similarity_losses.append(val_sim_loss)
        val_question_sims.append(val_q_sim)
        val_schema_sims.append(val_s_sim)
        
        print(f"Train Loss: {train_loss:.4f} (Contrastive: {train_cont_loss:.4f}, Similarity: {train_sim_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Contrastive: {val_cont_loss:.4f}, Similarity: {val_sim_loss:.4f})")
        print(f"Train Sims - Question: {train_q_sim:.4f}, Schema: {train_s_sim:.4f}")
        print(f"Val Sims - Question: {val_q_sim:.4f}, Schema: {val_s_sim:.4f}")
        print(f"LR: {trainer.scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'contrastive_loss': val_cont_loss,
                'similarity_loss': val_sim_loss
            }, save_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")
        
        trainer.scheduler.step()
    
    final_save_path = save_path.replace('.pth', '_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'epoch': num_epochs - 1,
        'loss': val_losses[-1],
        'contrastive_loss': val_contrastive_losses[-1],
        'similarity_loss': val_similarity_losses[-1]
    }, final_save_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Stage 2 Total Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(range(1, num_epochs + 1), train_contrastive_losses, label='Train Contrastive')
    plt.plot(range(1, num_epochs + 1), val_contrastive_losses, label='Val Contrastive')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('Contrastive Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(range(1, num_epochs + 1), train_similarity_losses, label='Train Similarity')
    plt.plot(range(1, num_epochs + 1), val_similarity_losses, label='Val Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Similarity Loss')
    plt.title('Similarity Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(range(1, num_epochs + 1), train_question_sims, label='Train Enhanced vs Question')
    plt.plot(range(1, num_epochs + 1), val_question_sims, label='Val Enhanced vs Question')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title('Enhanced vs Question Similarity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(range(1, num_epochs + 1), train_schema_sims, label='Train Enhanced vs Schema')
    plt.plot(range(1, num_epochs + 1), val_schema_sims, label='Val Enhanced vs Schema')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title('Enhanced vs Schema Similarity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage2_comprehensive_metrics.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_question_sims, label='Train Enhanced vs Question')
    plt.plot(range(1, num_epochs + 1), train_schema_sims, label='Train Enhanced vs Schema')
    plt.plot(range(1, num_epochs + 1), val_question_sims, label='Val Enhanced vs Question')
    plt.plot(range(1, num_epochs + 1), val_schema_sims, label='Val Enhanced vs Schema')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'cosine_similarity.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
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
    """Main function for Stage 2 training"""
    data_file = 'datasets/train.json'
    embeddings_file = 'embeddings/stage2_embeddings.pt'
    stage1_embeddings_file = './embeddings/schema_aware_embeddings.pt'
    flag_model_path = 'path/to/bge-large-en-v1.5'
    stage2_model_save_path = './SAR/models/best_contrastive_model.pth'
    output_dir = './training_plots/stage2'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_stage2_contrastive_model(
        data_file=data_file,
        embeddings_file=embeddings_file,
        stage1_embeddings_file=stage1_embeddings_file,
        flag_model_path=flag_model_path,
        num_epochs=20,
        batch_size=16,
        learning_rate=1e-4,
        temperature=0.07,
        device=device,
        save_path=stage2_model_save_path,
        output_dir=output_dir
    )
    
    print("Stage 2 training completed!")


if __name__ == "__main__":
    main_stage2()