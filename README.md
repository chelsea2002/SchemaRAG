# SchemaRAG

The datasets used in the experiments — Spider, BIRD, BEAVER, and Spider 2.0 — can be downloaded from the following sources:

Spider: [https://yale-lily.github.io/spider/](https://yale-lily.github.io/spider)

BIRD: https://bird-bench.github.io/

BEAVER: https://github.com/peterbaile/beaver

Spider 2.0: https://spider2-sql.github.io/

UniSQL: The UniSQL dataset is included directly in this GitHub repository.

You can download our trained SchemaLinker model using the following command:
```python
# Model Download
from modelscope import snapshot_download

model_dir = snapshot_download('TonyTANG11/SchemaLinker')
```
Additionally, schema-aware data and contrastive learning datasets can be downloaded from the following link:
https://drive.google.com/file/d/1tK-cK5y4G94_EMxzZnghl_aZhzoVi7DZ/view

🏗️ Architecture
SchemaRAG consists of three core components:

1. SchemaLinker

PromptSchema: Automatic schema interpretation with BM25S-based sampling
CoT-aligned Training: Knowledge distillation from high-quality GPT-4o rationales
Multi-task Alignment: Error detection, correction, and answer generation
GRPO Fine-tuning: Reinforcement learning for optimal schema element selection

2. Schema-Augmented Retriever (SAR)

Schema-Aware Embeddings: Cross-attention between question and database schema
Contrastive Learning: Enhanced discriminability of SQL syntactic structures
Structure-Focused Retrieval: Retrieves examples based on SQL syntax similarity, not just text

3. Pareto-Optimal SQL Generator (POSG)

Multi-Candidate Generation: Generates diverse SQL query candidates
Three-Dimensional Evaluation:

Executability (S_ex)
Schema linking conformity (S_sl)
Example consistency (S_ec)


Pareto Selection: Identifies non-dominated optimal queries

🤝 Contributing
We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
