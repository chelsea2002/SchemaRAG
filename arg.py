import os
import argparse

def main_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', default='bird_dev', type=str, help='dataset name')
    arg_parser.add_argument('--batch_size', default=64, type=int, help='batch size for preprocessing')
    arg_parser.add_argument('--device', default=0, type=int, help='gpu id (-1 represents cpu)')
    arg_parser.add_argument('--example', default=1, type=int, help='example')
    arg_parser.add_argument('--gpt', default='gpt-4.1-mini', type=str, help='GPT model')
    arg_parser.add_argument('--db_sample_limit', default=5, type=int, help='number of samples per column')
    args = arg_parser.parse_args()
    return args