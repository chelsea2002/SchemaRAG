import os

from util.example import Example


dataset = "spider"
DATASET_SCHEMA = f"./data/spider/tables.json"
DATASET = f"./data/spider/dev.json"
Example.configuration(dataset)
datasets = Example.load_dataset(dataset, 'dev')

OUTPUT_FILE = f"log/spider/model/predicted.sql"
Example.evaluator.accuracy(os.path.join(OUTPUT_FILE), datasets, os.path.join(f'log/spider/model/dev.txt'), etype="all")
