import os
import sys
import math
import time
import numpy as np
import random
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Data import *
from Model import *
from transformers import get_scheduler, SchedulerType, set_seed
import wandb

def update_counts(counts, b_data, b_scores, opts, data_obj, all_candidates=None):
    entity2id = data_obj.entity2id
    id2entity = data_obj.id2entity
    for scores, piece in zip(b_scores, b_data):
        # random.shuffle(scores)    # uncomment this line to build random baseline
        sorted_indexes = np.argsort(scores)[::-1]
        gold_idx = {entity2id[id] for id in piece["all_answer_entities"]}
        counts['n_data'] = counts['n_data'] + 1
        # std eval
        gold_id = piece["answer_entity_id"]
        gold_city = gold_id.split("_")[0]
        index = 1 if "_A_" in gold_id else 0 if "_R_" in gold_id else 2
        candidates = all_candidates[gold_city][index]
        std_candidate_idx = {entity2id[id] for id in candidates}
        std_sorted_indexes = [i for i in sorted_indexes if i in std_candidate_idx]
        for pos,idx in enumerate(std_sorted_indexes):
            if idx in gold_idx:
                break
        counts['std_mrr'] = counts['std_mrr'] + 1/(pos+1)
        counts['std_pos'].append(pos)
        if pos<3: counts['std_acc@3'] = counts['std_acc@3'] + 1
        if pos<5: counts['std_acc@5'] = counts['std_acc@5'] + 1
        if pos<30: counts['std_acc@30'] = counts['std_acc@30'] + 1
        if pos<100: counts['std_acc@100'] = counts['std_acc@100'] + 1

        # last checkpoint top 100 re-order eval
        for pos,idx in enumerate(sorted_indexes):
            if idx in gold_idx:
                break
        counts['mrr'] = counts['mrr'] + 1/(pos+1)
        if pos<3: counts['acc@3'] = counts['acc@3'] + 1
        if pos<5: counts['acc@5'] = counts['acc@5'] + 1
        if pos<30: counts['acc@30'] = counts['acc@30'] + 1
        if pos<100: counts['acc@100'] = counts['acc@100'] + 1

        if opts.test_mode: # save prediction for human eval
            piece['prediction'] = [id2entity[i] for i in std_sorted_indexes[:30]]

    return counts


def test(opts, data_obj, network, test_data, step=0):

    all_candidates = data_obj.get_all_candidates()
    data_obj.reload_entity_embeddings(opts.output_dir)
    network.eval()
    batch_size = opts.test_batch_size
    no_batches = math.ceil(len(test_data) / batch_size)
    logging.info(f"Running Test data: {len(test_data)}, with batch_size: {batch_size}, total batches: {no_batches}.")

    counts = {'n_data':0, 'acc@3':0, 'acc@5':0, 'acc@30':0, 'acc@100':0, 'std_acc@3':0, 'std_acc@5':0, 'std_acc@30':0, 'std_acc@100':0, 'mrr':0, 'std_mrr':0, 'std_pos':[],}
    for i in tqdm(range(no_batches)):
        with torch.no_grad():
            b_data = test_data[batch_size*i : batch_size*(i+1)]
            batch = data_obj.prepare_forward(b_data, training=False, device=opts.device)
            scores = network(**batch, training=False)

        scores = scores.detach().cpu().numpy()
        counts = update_counts(counts, b_data, scores, opts, data_obj, all_candidates)
    eval_results = {'acc@3': counts['acc@3'] / counts['n_data'] * 100,
                    'acc@5': counts['acc@5'] / counts['n_data'] * 100,
                    'acc@30': counts['acc@30'] / counts['n_data'] * 100,
                    'acc@100': counts['acc@100'] / counts['n_data'] * 100,
                    'mrr': counts['mrr'] / counts['n_data'],
                    'std_acc@3': counts['std_acc@3'] / counts['n_data'] * 100,
                    'std_acc@5': counts['std_acc@5'] / counts['n_data'] * 100,
                    'std_acc@30': counts['std_acc@30'] / counts['n_data'] * 100,
                    'std_acc@100': counts['std_acc@100'] / counts['n_data'] * 100,
                    'std_mrr': counts['std_mrr'] / counts['n_data'],
                    }

    if opts.use_wandb:
        wandb.log(eval_results)
    logging.info("----------")
    for k,v in eval_results.items():
        logging.info(f"Eval results on epoch-{step} -- {k}: {v}")
    logging.info("----------")

    with open(os.path.join(opts.output_dir,f'{opts.prefix}_{step}_res.json'),'w') as f:
        json.dump(counts, f)
    if opts.test_mode:
        for piece in test_data:
            piece.pop('all_answer_entities')
            if 'hard_neg' in piece:
                piece.pop('hard_neg')
        with open(os.path.join(opts.output_dir,f'{opts.prefix}_{step}_prediction.json'),'w') as f:
            json.dump(test_data, f)

def rebuild_hard_neg(opts, data_obj, network, training_data):
    logging.info(f"Rebuild training data with negatives.")
    network.eval()
    batch_size = opts.test_batch_size
    no_batches = math.ceil(len(training_data) / batch_size)
    all_candidates = data_obj.get_all_candidates()
    for i in tqdm(range(no_batches)):
        with torch.no_grad():
            b_data = training_data[batch_size*i : batch_size*(i+1)]
            batch = data_obj.prepare_forward(b_data, training=False, device=opts.device)
            scores = network(**batch, training=False)

        b_gold_ids = [x["all_answer_entities"] for x in training_data[batch_size*i : batch_size*(i+1)]]
        b_gold_id = [x["answer_entity_id"] for x in training_data[batch_size*i : batch_size*(i+1)]]
        b_scores = scores.detach().cpu().numpy()

        for scores, gold_ids, gold_id, piece in zip(b_scores, b_gold_ids, b_gold_id, b_data):
            hard_neg = []
            gold_idx = {data_obj.entity2id[id] for id in gold_ids}
            gold_city = gold_id.split("_")[0]
            index = 1 if "_A_" in gold_id else 0 if "_R_" in gold_id else 2
            candidates = all_candidates[gold_city][index]
            std_candidate_idx = {data_obj.entity2id[id] for id in candidates}
            sorted_indexes = np.argsort(scores)[::-1]
            for idx in sorted_indexes:
                if len(hard_neg) == 10 *opts.hard_negatives_per_qa:
                    break
                if idx not in gold_idx and idx in std_candidate_idx:
                    hard_neg.append(data_obj.id2entity[idx])
            piece['hard_neg'] = hard_neg

    return training_data


def train(opts, data_obj, network, training_data, test_data):
    device = opts.device
    batch_size = opts.batch_size
    if opts.loss.lower() == 'mr':
        criterion = nn.MarginRankingLoss(margin=opts.margin, reduction='mean')
    elif opts.loss.lower() == 'nll':
        criterion = nn.NLLLoss()
    else:
        logging.info("Not defined loss function.")

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in network.named_parameters()],
            "weight_decay": 0.0,
        },
        ]

    no_batches = math.ceil(len(training_data) / batch_size)

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=opts.lr)
    lr_scheduler = get_scheduler(
        name=opts.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=opts.num_warmup_steps,
        num_training_steps=opts.num_train_epochs * int(no_batches / opts.gradient_accumulation_steps),
    )

    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(training_data)}")
    logging.info(f"  Instantaneous batch size per device = {batch_size}")
    logging.info(f"  Gradient Accumulation steps = {opts.gradient_accumulation_steps}")
    log_loss = 0
    for epoch in range(opts.num_train_epochs):
        if epoch >= opts.s2_after:
            training_data = rebuild_hard_neg(opts, data_obj, network, training_data)
        expand_training_data = data_obj.expand_with_negatives(training_data, training=True)
        for i in range(no_batches):
            network.train()
            batch = expand_training_data[(i * batch_size) : (i+1) * batch_size]
            batch = data_obj.prepare_forward(batch, training=True, device=device)
            scores = network(**batch, training=True)

            if opts.loss.lower() == 'mr':
                y = torch.ones(scores[:,1:].size()).to(device)
                loss = criterion(scores[:,:1], scores[:,1:], y)
            elif opts.loss.lower() == 'nll':
                scores = F.log_softmax(scores, dim=1)
                target = torch.tensor([0]*scores.size(0)).to(device)
                loss = criterion(scores, target)
            else:
                raise ValueError(f'Loss not configured: {opts.loss.lower()}')

            loss = loss / opts.gradient_accumulation_steps
            loss.backward()
            log_loss += loss.item()
            if (i+1) % opts.gradient_accumulation_steps == 0:
                if opts.use_wandb:
                    wandb.log({'train_loss': log_loss,
                               'lr': lr_scheduler.get_last_lr()[-1]})
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                log_loss = 0

        # eval for each batch
        logging.info(f'Save for epoch {epoch}')
        torch.save(network.state_dict(), os.path.join(opts.output_dir,f'{opts.prefix}.weights'))
        network.save_entity_embeds(opts, data_obj)
        test(opts, data_obj, network, test_data, epoch)




def main():
    parser = argparse.ArgumentParser(description='TourQue Executor')

    # training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--loss', type=str, default="nll")
    parser.add_argument("--margin", type=float, default=1.0) # if use margin ranking loss
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_train_epochs", type=int, default=12)
    parser.add_argument("--s2_after", type=int, default=8)
    # contrastive
    parser.add_argument("--samples_per_qa", type=int, default=9)
    parser.add_argument("--hard_negatives_per_qa", type=int, default=8)
    parser.add_argument("--score_method", type=str, default="dot", choices=['dot','cosine'])
    # textual module
    parser.add_argument("--q_encoder", type=str, default='distilbert-base-uncased')
    parser.add_argument("--e_encoder", type=str, default='distilbert-base-uncased')
    parser.add_argument("--l_encoder", type=str, default='distilbert-base-uncased')
    parser.add_argument("--max_question_length", type=int, default=384)
    parser.add_argument("--max_entity_length", type=int, default=512)
    parser.add_argument("--n_cluster_reviews", type=int, default=0)
    parser.add_argument("--encode_entity_name", action="store_true")
    parser.add_argument("--encoder_out_dim", type=int, default=256)
    # location module
    parser.add_argument("--loc", type=str, default="none", choices=['none','text','numeric'])
    parser.add_argument("--max_loc_length", type=int, default=64)
    parser.add_argument("--loc_out_dim", type=int, default=64)
    parser.add_argument("--max_locs", type=int, default=5)
    # distance module
    parser.add_argument("--distance", action="store_true")
    parser.add_argument("--haversine_distance", action="store_true")
    parser.add_argument("--dist_weight", type=float, default=0.2)
    # test
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--batch_size_save_entity", type=int, default=24)
    # other
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--data_dir', type=str, default="../data/")
    parser.add_argument('--test_file', type=str, default="test.json")
    parser.add_argument('--train_file', type=str, default="train.json")
    parser.add_argument('--valid_file', type=str, default="valid.json")
    parser.add_argument("--id_file", type=str,default="entity_id.json")
    parser.add_argument("--emb_file", type=str,default="entity_emb.np")
    parser.add_argument("--knowledge_file", type=str,default="TourQue_Knowledge_Cluster.json")
    parser.add_argument("--cities_lat_long_file", type=str, default="final_cities_lat_long.json")

    opts = parser.parse_args(sys.argv[1:])

    set_seed(opts.seed)
    opts.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(str(opts))

    os.makedirs(opts.output_dir, exist_ok=True)
    import base64
    opts.prefix = \
    f'LOG_{opts.samples_per_qa}_{opts.hard_negatives_per_qa}_{opts.batch_size*opts.gradient_accumulation_steps}_{opts.lr}_{opts.num_train_epochs}_{opts.s2_after}_{opts.score_method}_{opts.loss}_{opts.encode_entity_name}_{opts.loc}_{opts.max_locs}_{opts.distance}_{opts.dist_weight}_{opts.haversine_distance}_{opts.seed}_{opts.n_cluster_reviews}_{opts.train_file}_{opts.test_file}_{opts.q_encoder.split("/")[-1]}_{opts.e_encoder.split("/")[-1]}_{opts.knowledge_file}_'
    opts.id_file = opts.prefix + opts.id_file
    opts.emb_file = opts.prefix + opts.emb_file

    if opts.use_wandb:
        wandb.init(project="new_tourqa", config=opts)
    log_file = os.path.join(opts.output_dir, opts.prefix + ".log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

    # Load Data and Model
    data_obj = Lamb_Data(opts=opts)
    network = Lamb(opts)
    # wandb.watch(
    #         network, criterion=None, log="all", log_freq=2000, idx=None,
    #             log_graph=(False)
    #         )

    test_file = os.path.join(opts.data_dir, 'TourQue_QA_Pairs', opts.test_file)
    train_file = os.path.join(opts.data_dir, 'TourQue_QA_Pairs', opts.train_file)
    valid_file = os.path.join(opts.data_dir, 'TourQue_QA_Pairs', opts.valid_file)

    test_data = data_obj.load_data_from_file(test_file, training=False, debug=opts.debug)
    if opts.test_mode:
        network.load_state_dict(torch.load(os.path.join(opts.output_dir,f'{opts.prefix}.weights')))
        network.to(opts.device)
        test(opts, data_obj, network, test_data)
    else:
        network.to(opts.device)
        training_data = data_obj.load_data_from_file(train_file, training=True, debug=opts.debug)
        # DEBUG
        # training_data = training_data[:100]
        # DEBUG-end
        train(opts, data_obj, network, training_data, test_data)


if __name__ == "__main__":
    main()
