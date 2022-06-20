import logging
import os
import tqdm
import random
import sys
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional
from argparse import Namespace
from torch.utils.data import DataLoader
import numpy as np

from sklearn.cluster import KMeans

import transformers
from transformers import (
    AutoConfig,
    BertModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/")
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-uncased')
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args(sys.argv[1:])

    import json
    with open(os.path.join(args.data_dir, 'TourQue_Knowledge_Cluster.json')) as f:
        knowledge = json.load(f)

    args.tmp_dir = os.path.join(args.data_dir, 'tmp')
    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)


    if os.path.exists(os.path.join(args.tmp_dir, 'id2reviews.json')):
        with open(os.path.join(args.tmp_dir, 'all_embeddings.np'),'rb') as f:
            all_embeddings = np.load(f)
        with open(os.path.join(args.tmp_dir, 'id2reviews.json'),'r') as f:
            id2reviews = json.load(f)
    else:

        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            output_hidden_states=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
        )

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

        model = BertModel.from_pretrained(args.model_name_or_path,config=config)
        model.to(device)

        all_reviews = []
        id2reviews = {}
        i = 0
        for k,v in knowledge.items():
            for sent in v['review']:
                all_reviews.append(sent)
            if i != len(all_reviews):
                id2reviews[k] = [x for x in range(i,len(all_reviews))]
            i = len(all_reviews)
        print(len(all_reviews))

        def tokenize(batch):
            return tokenizer(batch, padding="max_length", max_length=args.max_length, truncation=True, return_tensors='pt')
        eval_dataloader = DataLoader(all_reviews, batch_size=args.batch_size, collate_fn=tokenize, num_workers=4)

        # generate embeddings
        all_embeddings = []
        model.eval()
        for batch in tqdm.tqdm(eval_dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            all_embeddings.append(outputs.pooler_output.detach().cpu().numpy())
        all_embeddings = np.concatenate(all_embeddings)

        print("Save embeddings.")
        with open(os.path.join(args.tmp_dir, 'all_embeddings.np'),'wb') as f:
            np.save(f,all_embeddings)
        with open(os.path.join(args.tmp_dir, 'id2reviews.json'),'w') as f:
            json.dump(id2reviews, f)


    # clustering
    entity_ids = list(id2reviews.keys())
    for k in tqdm.tqdm(entity_ids):
        v = id2reviews[k]
        entity_embeddings = all_embeddings[v[0]:v[-1]+1]
        assert len(entity_embeddings) == len(v)
        cluster_map = {}
        for n in range(3,10,2):
            if len(entity_embeddings) > n:
                cluster = KMeans(n_clusters=n, random_state=0).fit(entity_embeddings).labels_
                cluster_map[n] = cluster.tolist()
        knowledge[k]['cluster_map'] = cluster_map

    with open(os.path.join(args.data_dir, 'TourQue_Knowledge_Cluster.json'),'w') as f:
        json.dump(knowledge,f,indent=2)

main()
