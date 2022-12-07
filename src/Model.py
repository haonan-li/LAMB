import os
import sys
import copy
import time
import json
import random
import numpy as np
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


class DistanceEncoder(nn.Module):
    def __init__(self, options):
        super(DistanceEncoder, self).__init__()
        self.opts = options
        self.max_loc = self.opts.max_locations
        self.samples_per_qa = options.samples_per_qa
        # self.dropout = nn.Dropout(0.1)
        # self.fc1 = nn.Linear(self.max_loc, 2*self.max_loc)
        # self.fc2 = nn.Linear(2*self.max_loc, 2*self.max_loc)
        # self.fc3 = nn.Linear(2*self.max_loc, 1)

    def forward(self, question_latlongs=None, question_latlong_mask=None, candidate_latlongs=None, training=True):
        max_loc = self.max_loc
        B = question_latlong_mask.size(0)
        if training:
            sitr = self.samples_per_qa
        else:
            sitr = candidate_latlongs.size(0)
            candidate_latlongs = candidate_latlongs.unsqueeze(0).expand(B, sitr, 2)

        dist = 2 * torch.ones((B, sitr, max_loc)).to(self.opts.device)
        for i in range(B):
            m = question_latlong_mask[i]    # (1, max_loc)
            if not torch.any(m):    # no tagged latlong in q
                continue
            m = m.repeat(sitr,1)    # (sitr, max_loc)
            q = question_latlongs[i] * torch.tensor([90,180]).to(self.opts.device)   # (max_loc, 2)
            q = q.unsqueeze(0).expand(sitr,max_loc,2)    # (sitr, max_loc, 2)
            c = candidate_latlongs[i].unsqueeze(1) * torch.tensor([90,180]).to(self.opts.device)   # (sitr, 1, 2)
            if self.opts.haversine_distance: # Haversine formula, range about [0, 1.6]
                q = torch.deg2rad(q)
                c = torch.deg2rad(c)

                cos_c = torch.cos(c)
                cos_q = torch.cos(q)

                diff = q-c

                a = torch.sin(diff[:,:,0]/2)**2 + (cos_c[:,:,0] * cos_q[:,:,0]) * torch.sin(diff[:,:,1]/2)**2
                dis = torch.asin(torch.sqrt(a))
            else:
                dis = torch.sum(torch.abs(q-c), -1)    # (sitr, max_loc)
            masked_dis = dis.masked_fill_(~m, 2)       # (sitr, max_loc)
            dist[i] = masked_dis
        min_x = - torch.min(dist, axis=-1).values.unsqueeze(-1)
        return min_x


class LocationEncoder(nn.Module):
    def __init__(self, options, is_entity=True):
        super(LocationEncoder, self).__init__()
        self.opts = options
        self.max_loc = self.opts.max_locations
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


class Lamb(nn.Module):
    def __init__(self, options):
        super(Lamb, self).__init__()
        self.opts = options
        self.batch_size = options.batch_size
        self.samples_per_qa = options.samples_per_qa
        self.max_loc = self.opts.max_locations
        self.dropout = nn.Dropout(0.2)
        self.encoder_output_dim = self.opts.encoder_output_dim
        self.relu = nn.ReLU()

        if self.opts.location:
            self.output_dim = self.encoder_output_dim + 2*self.max_loc
        else:
            self.output_dim = self.encoder_output_dim

        if self.opts.transformer_model == 'sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco':
            self.transformer_config = AutoConfig.from_pretrained('distilbert-base-uncased')
            self.hidden_dim = self.transformer_config.compression_dim
        if self.opts.transformer_model == 'huawei-noah/TinyBERT_General_4L_312D':
            self.transformer_config = AutoConfig.from_pretrained('bert-base-uncased')
            self.hidden_dim = self.transformer_config.hidden_size
        else:
            self.transformer_config = AutoConfig.from_pretrained(self.opts.transformer_model)
            self.hidden_dim = self.transformer_config.hidden_size

        # textual encoder
        if self.opts.debug: # create a simple model for debugging
            self.question_encoder = lambda input_ids, token_type_ids=None, attention_mask=None: (torch.randn((input_ids.size(0), 1, self.hidden_dim)).to(self.opts.device), 0)
            self.entity_encoder = lambda input_ids, token_type_ids=None, attention_mask=None: (torch.randn((input_ids.size(0), 1, self.hidden_dim)).to(self.opts.device), 0)
        else:
            self.question_encoder = AutoModel.from_pretrained(self.opts.transformer_model, from_tf=False, config=self.transformer_config)
            self.entity_encoder = AutoModel.from_pretrained(self.opts.transformer_model, from_tf=False, config=self.transformer_config)

        # output layer
        self.question_output_layer = nn.Linear(self.hidden_dim, self.encoder_output_dim)
        self.entity_output_layer = nn.Linear(self.hidden_dim, self.encoder_output_dim)

        # fuse layer
        self.question_fuse_layer = nn.Linear(self.output_dim, self.output_dim)
        self.entity_fuse_layer = nn.Linear(self.output_dim, self.output_dim)

        self.scale_factor_regular = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Tanh())
        self.dist_scale_factor_regular = nn.Sequential(nn.Tanh())

        if self.opts.distance:
            self.distance_encoder = DistanceEncoder(options)
        if self.opts.location:
            self.q_location_encoder = LocationEncoder(self.opts, is_entity=False)
            self.e_location_encoder = LocationEncoder(self.opts, is_entity=True)

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
            return torch.bmm(question_embeddings.view(B, 1, self.output_dim),
                             entity_embeddings.view(B, self.output_dim, 1))


    def forward(self, question_inputs, question_latlongs=None, question_latlong_mask=None,
                candidate_inputs=None, candidate_latlongs=None, candidate_entity_vectors=None, training=True):
        # questions_inputs['input_ids']:  (B, max_length)
        # candidate_inputs['input_ids']: (B*sitr, max_length)
        # question_latlongs:  (B, max_loc, 2)
        # candidate_latlongs: (B*sitr, 2)

        B = question_inputs['input_ids'].size(0)
        sitr = self.samples_per_qa if training else candidate_entity_vectors.size(0) // B

        # Encode question
        hidden_state_q = self.question_encoder(**question_inputs)[0]
        pooled_output_q = hidden_state_q[:, 0]
        output_q = self.question_output_layer(self.relu(self.dropout(pooled_output_q)))
        if training:
            output_q = output_q.unsqueeze(1).expand(B,sitr,self.encoder_output_dim).reshape(B*sitr, self.encoder_output_dim)

        # Encode canddidates
        if training:
            hidden_state_c = self.entity_encoder(**candidate_inputs)[0]
            pooled_output_c = hidden_state_c[:, 0]
            output_c = self.entity_output_layer(self.relu(self.dropout(pooled_output_c)))
        else:
            output_c = candidate_entity_vectors

        # Location encoding
        if self.opts.location:
            output_loc_q = self.q_location_encoder(question_latlongs, question_latlong_mask)
            dim_loc_q = output_loc_q.size(-1)
            if training:
                output_loc_q = output_loc_q.unsqueeze(1).expand(B,sitr,dim_loc_q).reshape(B*sitr, dim_loc_q)
            output_q = torch.cat((output_q, output_loc_q), dim=-1)
            if training:
                output_loc_c = self.e_location_encoder(candidate_latlongs)
                output_c = torch.cat((output_c, output_loc_c), dim=-1)

        # Fusing
        output_q = self.question_fuse_layer(output_q) # (B*sitr, output_dim)
        output_c = self.entity_fuse_layer(output_c)   # (B*sitr, output_dim)
        # Compute scores
        scores = self.compute_score(output_q, output_c, training).reshape(B,-1) # (B*sitr)

        # Encode distance (only when not training)
        if not training and self.opts.distance:
            dist_scores = self.distance_encoder(question_latlongs, question_latlong_mask, candidate_latlongs, training)
            # dist_scores = dist_scores.reshape(B, -1)
            dist_scores = self.dist_scale_factor_regular(dist_scores).reshape(B, -1)
            assert scores.size() == dist_scores.size()
            scores = (1-self.opts.dist_weight) * scores + self.opts.dist_weight * dist_scores

        return scores

    def save_entity_embeds(self, opts, data_obj):
        device = opts.device
        all_entity_emb = []
        all_entity_id = data_obj.entity_ids
        entity_dataloader = DataLoader(data_obj.processed_entities, collate_fn=default_data_collator, batch_size=self.opts.batch_size_save_entity)
        self.eval()
        with torch.no_grad():
            for batch in tqdm(entity_dataloader):
                t_batch = {k:v.to(device) for k, v in batch.items() if k != 'latlongs'}
                l_batch = {k:v.to(device) for k, v in batch.items() if k == 'latlongs'}
                hidden_state_c = self.entity_encoder(**t_batch)[0]
                output_c = hidden_state_c[:, 0]
                output_c = self.entity_output_layer(output_c) # add output layer here, no need to do it when in eval forward.
                if self.opts.location:
                    output_loc_c = self.e_location_encoder(**l_batch)
                    output_c = torch.cat((output_c, output_loc_c), dim=-1)
                all_entity_emb.append(output_c.detach().cpu().numpy())
            all_entity_emb = np.concatenate(all_entity_emb)

        with open(os.path.join(opts.output_dir, opts.id_file),'w') as f:
            json.dump(all_entity_id, f)
        with open(os.path.join(opts.output_dir, opts.emb_file),'wb') as f:
            np.save(f,all_entity_emb)
