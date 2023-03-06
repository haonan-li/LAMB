import os
import sys
import copy
import time
import json
import random
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from Data import *
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

class Criterion(nn.Module):
    def __init__(self, options):
        super(Criterion, self).__init__()
        self.opts = options

        if self.opts.loss.lower() == 'mr':
            self.loss_func = nn.MarginRankingLoss(margin=self.opts.margin, reduction='mean')
        elif self.opts.loss.lower() == 'nll':
            self.loss_func = nn.NLLLoss()
        else:
            logging.info("Not defined loss function.")

    def forward(self, scores):
        if self.opts.loss.lower() == 'mr':
            y = torch.ones(scores[:,1:].size()).to(self.opts.device)
            loss = self.loss_func(scores[:,:1], scores[:,1:], y)
        elif self.opts.loss.lower() == 'nll':
            scores = F.log_softmax(scores, dim=1)
            target = torch.tensor([0]*scores.size(0)).to(self.opts.device)
            loss = self.loss_func(scores, target)
        return loss



class NumLocationEncoder(nn.Module):
    def __init__(self, options, is_entity=True):
        super(LocationEncoder, self).__init__()
        self.opts = options
        self.max_loc = self.opts.max_locs
        self.is_entity = is_entity
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        if is_entity:
            self.fc1 = nn.Linear(2, self.max_loc)
            self.fc2 = nn.Linear(self.max_loc, 2*self.max_loc)
            self.fc3 = nn.Linear(2*self.max_loc, 2*self.max_loc)
        else:
            self.fc1 = nn.Linear(2*self.max_loc, 2*self.max_loc)
            self.fc2 = nn.Linear(2*self.max_loc, 2*self.max_loc)
            self.fc3 = nn.Linear(2*self.max_loc, 2*self.max_loc)

    def forward(self, latlongs, latlong_mask=None):
        latlongs = latlongs.float()
        if not self.is_entity:
            latlongs = latlongs.reshape(latlongs.size(0),-1)
        x = self.dropout(self.relu(self.fc1(latlongs)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class TextLocationEncoder(nn.Module):
    def __init__(self, options, is_entity=True):
        super(TextLocationEncoder, self).__init__()
        self.opts = options
        self.l_config = AutoConfig.from_pretrained(self.opts.l_encoder)
        self.l_hidden_dim = self.l_config.hidden_size
        self.loc_encoder = AutoModel.from_pretrained(self.opts.l_encoder, from_tf=False, config=self.l_config)
        self.loc_encoder.transformer.layer = self.loc_encoder.transformer.layer[:self.opts.n_loc_layers] # only keep first N layers
        self.loc_out_layer = nn.Linear(self.l_hidden_dim, self.opts.loc_out_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, entity_loc_inputs):
        hidden_state_e_loc = self.loc_encoder(**entity_loc_inputs)[0]
        pooled_out_e_loc = hidden_state_e_loc[:, 0]
        out_e_loc = self.loc_out_layer(self.relu(self.dropout(pooled_out_e_loc)))
        return out_e_loc

class Lamb(nn.Module):
    def __init__(self, options):
        super(Lamb, self).__init__()
        self.opts = options
        self.batch_size = options.batch_size
        self.samples_per_qa = options.samples_per_qa
        self.max_loc = self.opts.max_locs
        self.dropout = nn.Dropout(0.2)
        self.encoder_out_dim = self.opts.encoder_out_dim
        self.loc_out_dim= self.opts.loc_out_dim
        self.relu = nn.ReLU()

        if self.opts.loc == 'numeric':
            self.out_dim = self.encoder_out_dim + 2*self.max_loc
        elif self.opts.loc == 'text':
            self.out_dim = self.encoder_out_dim + self.loc_out_dim
        else:
            self.out_dim = self.encoder_out_dim

        # q_encoder init
        self.q_config = AutoConfig.from_pretrained(self.opts.q_encoder)
        self.q_hidden_dim = self.q_config.hidden_size


        # e_encoder init
        self.e_config = AutoConfig.from_pretrained(self.opts.e_encoder)
        self.e_hidden_dim = self.e_config.hidden_size

        # DEBUG: textual encoder
        if self.opts.debug: # create a simple model for debugging
            self.question_encoder = lambda input_ids, token_type_ids=None, attention_mask=None: (torch.randn((input_ids.size(0), 1, self.hidden_dim)).to(self.opts.device), 0)
            self.entity_encoder = lambda input_ids, token_type_ids=None, attention_mask=None: (torch.randn((input_ids.size(0), 1, self.hidden_dim)).to(self.opts.device), 0)
        else:
            self.question_encoder = AutoModel.from_pretrained(self.opts.q_encoder, from_tf=False, config=self.q_config)
            self.entity_encoder = AutoModel.from_pretrained(self.opts.e_encoder, from_tf=False, config=self.e_config)

        # output layer
        self.question_out_layer = nn.Linear(self.q_hidden_dim, self.encoder_out_dim)
        self.entity_out_layer = nn.Linear(self.e_hidden_dim, self.encoder_out_dim)

        # fuse layer
        self.question_fuse_layer = nn.Linear(self.out_dim, self.out_dim)
        self.entity_fuse_layer = nn.Linear(self.out_dim, self.out_dim)

        self.scale_factor_regular = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Tanh())

        if self.opts.loc == 'numeric':
            self.q_loc_encoder = LocationcEncoder(self.opts, is_entity=False)
            self.e_loc_encoder = LocationEncoder(self.opts, is_entity=True)
        if self.opts.loc == 'text':
            self.q_loc_encoder = nn.Linear(self.q_hidden_dim, self.loc_out_dim)
            self.e_loc_encoder = TextLocationEncoder(self.opts)

        print(self)

    def compute_score(self, question_embeddings, entity_embeddings, training=True):
        if not training:
            if self.opts.score_method == 'dot':
                return torch.mm(question_embeddings, entity_embeddings.T).unsqueeze(-1)
            elif self.opts.score_method == 'cosine':
                return F.cosine_similarity(question_embeddings.unsqueeze(1), entity_embeddings.unsqueeze(0), dim=2)
        B = entity_embeddings.size(0)
        if self.opts.score_method == 'cosine':
            # times hidden_size, so no need to adjust other hyper-parameters
            return F.cosine_similarity(question_embeddings, entity_embeddings).unsqueeze(-1) * question_embeddings.size(-1)
        elif self.opts.score_method == 'dot':
            return torch.bmm(question_embeddings.view(B, 1, self.out_dim),
                             entity_embeddings.view(B, self.out_dim, 1))


    def forward(self, question_inputs, question_latlongs=None, question_latlong_mask=None,
                entity_inputs=None, entity_loc_inputs=None, entity_latlongs=None, entity_vectors=None, training=True):
        # questions_inputs['input_ids']:  (B, max_length)
        # entity_inputs['input_ids']: (B*sitr, max_length)
        # question_latlongs:  (B, max_loc, 2)
        # entity_latlongs: (B*sitr, 2)

        B = question_inputs['input_ids'].size(0)
        sitr = self.samples_per_qa if training else entity_vectors.size(0) // B

        # Encode question
        hidden_state_q = self.question_encoder(**question_inputs)[0]
        pooled_out_q = hidden_state_q[:, 0]
        out_q = self.question_out_layer(self.relu(self.dropout(pooled_out_q)))
        if training:
            out_q = out_q.unsqueeze(1).expand(B,sitr,self.encoder_out_dim).reshape(B*sitr, self.encoder_out_dim)

        # Encode canddidates
        if training:
            hidden_state_c = self.entity_encoder(**entity_inputs)[0]
            pooled_out_c = hidden_state_c[:, 0]
            out_c = self.entity_out_layer(self.relu(self.dropout(pooled_out_c)))
        else:
            out_c = entity_vectors

        # Location encoding
        if self.opts.loc == 'numeric':
            out_loc_q = self.q_loc_encoder(question_latlongs, question_latlong_mask)
            dim_loc_q = out_loc_q.size(-1)
            if training:
                out_loc_q = out_loc_q.unsqueeze(1).expand(B,sitr,dim_loc_q).reshape(B*sitr, dim_loc_q)
            out_q = torch.cat((out_q, out_loc_q), dim=-1)
            if training:
                out_loc_c = self.e_loc_encoder(entity_latlongs)
                out_c = torch.cat((out_c, out_loc_c), dim=-1)
        if self.opts.loc == 'text':
            out_loc_q = self.q_loc_encoder(self.relu(self.dropout(pooled_out_q)))
            if training:
                out_loc_q = out_loc_q.unsqueeze(1).expand(B,sitr,self.loc_out_dim).reshape(B*sitr, self.loc_out_dim)
            out_q = torch.cat((out_q, out_loc_q), dim=-1)
            if training:
                out_loc_c = self.e_loc_encoder(entity_loc_inputs)
                out_c = torch.cat((out_c, out_loc_c), dim=-1)

        # Fusing
        out_q = self.question_fuse_layer(out_q) # (B*sitr, out_dim)
        out_c = self.entity_fuse_layer(out_c)   # (B*sitr, out_dim)
        # Compute scores
        scores = self.compute_score(out_q, out_c, training).reshape(B,-1) # (B*sitr)

        return scores

    def save_entity_embeds(self, opts, data_obj):
        device = opts.device
        all_entity_emb = []
        all_entity_id = data_obj.entity_ids
        entity_dataloader = DataLoader(data_obj.processed_entities, collate_fn=default_data_collator, batch_size=self.opts.batch_size_save_entity)
        self.eval()
        with torch.no_grad():
            for batch in tqdm(entity_dataloader):
                t_batch = {k:v.to(device) for k, v in batch.items() if (k != 'latlongs' and k[:4] != 'loc_')}
                l_num_batch = {k:v.to(device) for k, v in batch.items() if k == 'latlongs'}
                l_text_batch = {k[4:]:v.to(device) for k, v in batch.items() if k[:4] == 'loc_'}
                hidden_state_c = self.entity_encoder(**t_batch)[0]
                out_c = hidden_state_c[:, 0]
                out_c = self.entity_out_layer(out_c) # add out layer here, no need to do it when in eval forward.
                if self.opts.loc == 'numeric':
                    out_loc_c = self.e_loc_encoder(**l_num_batch)
                    out_c = torch.cat((out_c, out_loc_c), dim=-1)
                elif self.opts.loc == 'text':
                    out_loc_c = self.e_loc_encoder(l_text_batch)
                    out_c = torch.cat((out_c, out_loc_c), dim=-1)
                all_entity_emb.append(out_c.detach().cpu().numpy())
            all_entity_emb = np.concatenate(all_entity_emb)

        with open(os.path.join(opts.output_dir, opts.id_file),'w') as f:
            json.dump(all_entity_id, f)
        with open(os.path.join(opts.output_dir, opts.emb_file),'wb') as f:
            np.save(f,all_entity_emb)
