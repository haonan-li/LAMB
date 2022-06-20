import logging
import os
import tqdm
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional
from argparse import Namespace
from torch.utils.data import DataLoader

import numpy as np

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

args = Namespace(
    model_name_or_path='bert-base-uncased',
    config_name=None,
    tokenizer_name=None,
    model_revision='main',
    cache_dir=None,
    use_auth_token=False,
    use_fast_tokenizer=True,
    max_length=128,
    batch_size=32,
)


config = AutoConfig.from_pretrained(
    model_args.config_name if args.config_name else args.model_name_or_path,
    cache_dir=args.cache_dir,
    revision=args.model_revision,
    output_hidden_states=True,
    use_auth_token=True if args.use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    cache_dir=args.cache_dir,
    use_fast=args.use_fast_tokenizer,
    revision=args.model_revision,
    use_auth_token=True if args.use_auth_token else None,
)
# model = AutoModelForSequenceClassification.from_pretrained(
#     args.model_name_or_path,
#     from_tf=bool(".ckpt" in args.model_name_or_path),
#     config=config,
#     cache_dir=args.cache_dir,
#     revision=args.model_revision,
#     use_auth_token=True if args.use_auth_token else None,
# )

device = torch.device('cuda:0')

model = BertModel.from_pretrained(args.model_name_or_path,config=config)
model.to(device)

import json
with open('../data/TourQue_Knowledge.json') as f:
    reviews = json.load(f)

raw_data = []
id2reviews = {}
i = 0
for k,v in reviews.items():
    for sent in v['review']:
        raw_data.append(sent)
    if i != len(raw_data):
        id2reviews[k] = range(i,len(raw_data))
    i = len(raw_data)
print(len(raw_data))

def tokenize(batch):
    return tokenizer(batch, padding="max_length", max_length=args.max_length, truncation=True, return_tensors='pt')
eval_dataloader = DataLoader(raw_data, batch_size=args.batch_size, collate_fn=tokenize, num_workers=4)

review_embeddings = []
model.eval()
for batch in tqdm.tqdm(eval_dataloader):
    batch = {k:v.to(device) for k,v in batch.items()}
    outputs = model(**batch)
    review_embeddings.append(outputs.pooler_output.detach().cpu().numpy())
all_reviews = np.concatenate(review_embeddings)

entity_id = list(id2reviews.keys())
entity_emb = []
for k in entity_id:
    v = id2reviews[k]
    assert v[-1] <= len(all_reviews)
    entity_emb.append(np.mean(all_reviews[v[0]:v[-1]], axis=0))
entity_emb = np.stack(entity_emb)

with open('../data/entity_emb.np','wb') as f:
    np.save(f,entity_emb)
with open('../data/entity_id.json','w') as f:
    json.dump(entity_id, f)
