# SchemaRAG

he datasets used in the experiments — Spider, BIRD, BEAVER, and Spider 2.0 — can be downloaded from the following sources:

Spider: https://yale-lily.github.io/spider/

BIRD: https://bird-bench.github.io/

BEAVER: https://github.com/peterbaile/beaver

Spider 2.0: https://spider2-sql.github.io/

You can download our trained SchemaLinker model using the following command:
```python
# Model Download
from modelscope import snapshot_download

model_dir = snapshot_download('TonyTANG11/SchemaLinker')
```
