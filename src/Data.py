import os, sys, time
import json, re, copy
from tqdm import tqdm
from itertools import zip_longest
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


class Lamb_Data:
    def __init__(self, opts=None):
        self.opts=opts
        self.samples_per_qa = opts.samples_per_qa
        self.hard_negatives_per_qa = opts.hard_negatives_per_qa
        self.default_coordinate = {}
        self.entity_embeddings = None

        self.load_raw_review()
        self.load_city_coordinate()

        self.q_tokenizer = AutoTokenizer.from_pretrained(self.opts.q_encoder, use_fast=True)
        self.e_tokenizer = AutoTokenizer.from_pretrained(self.opts.e_encoder, use_fast=True)
        self.l_tokenizer = AutoTokenizer.from_pretrained(self.opts.e_encoder, use_fast=True) # use the same tokenizer as entity reviews
        self.cached_questions = {}

        self.preprocess_all_entities()


    def load_city_coordinate(self):
        with open(os.path.join(self.opts.data_dir, self.opts.cities_lat_long_file),'r') as f:
            for entry in json.load(f):
                self.default_coordinate[str(entry["city_id"])] = [entry["location"]["lat"], entry["location"]["lng"]]
            print(self.default_coordinate)


    def load_raw_review(self):
        with open(os.path.join(self.opts.data_dir, self.opts.knowledge_file),'r') as f:
            self.entity_knowledge = json.load(f)
            self.entity_ids = list(self.entity_knowledge)
            self.entity_set = set(self.entity_ids)
            self.entity2id = {entity:i for i,entity in enumerate(self.entity_ids)}
            self.id2entity = {i:entity for i,entity in enumerate(self.entity_ids)}
        print(f'Load entities knowledge total: {len(self.entity_knowledge)}.')

        if self.opts.n_cluster_reviews == 0:
            print(f'Keep reviews unchanged.')
            return
        if self.opts.n_cluster_reviews == 1:
            print(f'Shuffle reviews.')
            for k, v in tqdm(self.entity_knowledge.items()):
                random.shuffle(v['review'])
            return

        print(f'Reorder reviews according to cluster.')
        # reorder reviews by clustering
        for k, v in tqdm(self.entity_knowledge.items()):
            n = self.opts.n_cluster_reviews if str(self.opts.n_cluster_reviews) in v['cluster_map'] else 3
            cluster = np.array(v['cluster_map'][str(n)])
            review = [np.array(v['review'])[cluster==i] for i in range(n)]
            new_review = review[0]
            v['review'] = new_review


    def load_data_from_file(self, data_file, debug=False, training=True):
        print(f"Loading data from file: {data_file}")
        all_data = json.load(open(data_file, 'r'))
        if debug and training:
            print(f'In debug mode, keep 100 data points')
            all_data = all_data[:1000]
        qa_pairs = []
        # create qa pairs
        question_seen = set()
        for entry in tqdm(all_data):
            question = entry["question"]
            answer = entry["answer_entity_id"]
            if (not training and question in question_seen) or (not answer in self.entity_set):
                continue
            question_seen.add(question)
            obj = {}
            obj["question"] = entry["question"]
            obj["answer_entity_id"] = entry["answer_entity_id"]
            obj["all_answer_entities"] = set(filter(lambda x: x in self.entity_set, entry["all_answer_entities"]))
            obj["all_answer_entity_list"] = sorted(list(obj["all_answer_entities"]))
            if self.opts.loc == 'numeric':
                tagged_latlong = [(x[0],x[1]) for x in entry["tagged_locations_lat_long"]]
                obj["tagged_latlongs"] = list(set(tagged_latlong)) # unique location
            qa_pairs.append(obj)
        return qa_pairs


    def get_all_candidates(self):
        all_candidates = {}
        for entity in self.entity_set:
            city_id, entity_type, entity_num = entity.split('_')
            if city_id in all_candidates:
                if entity_type == 'A':
                    all_candidates[city_id][1].add(entity)
                else:
                    all_candidates[city_id][0].add(entity)
                fake_entity = city_id + "_H_" + entity_num
                all_candidates[city_id][2].add(entity)
            else:
                all_candidates[city_id] = [set(), set(), set()]
        all_candidates = {k:[sorted(list(x)) for x in v] for k,v in all_candidates.items()}
        return all_candidates


    def expand_with_negatives(self, data, training=True):
        if not training:
            return data

        all_candidates = self.get_all_candidates()
        samples = []
        print('Expand samples with candidates.')
        random.shuffle(data)
        for idx, p in enumerate(data):
            if idx % 1000 == 0:
                print(f"Build data amount: {idx}.")
            sample_obj = copy.deepcopy(p)
            answer_ent = p["answer_entity_id"]
            all_answer_entities = p["all_answer_entities"] # Type: set
            city = answer_ent.split("_")[0]
            index = 1 if "_A_" in answer_ent else  0 if "_R_" in answer_ent else 2
            candidate_pool = all_candidates[city][index]

            if 'hard_neg' in p:
                hard_negatives = random.sample(p['hard_neg'], k=min(len(p['hard_neg']), self.hard_negatives_per_qa))
            else:
                hard_negatives = random.sample(candidate_pool, k=min(len(candidate_pool), self.hard_negatives_per_qa+len(all_answer_entities)))
                hard_negatives = list(filter(lambda x: not x in all_answer_entities, hard_negatives))[:self.hard_negatives_per_qa] # remove positive
            easy_negatives = random.sample(self.entity_ids, k=self.samples_per_qa - len(hard_negatives) + len(all_answer_entities))
            easy_negatives = list(filter(lambda x: not x in all_answer_entities, easy_negatives))[:self.samples_per_qa-len(hard_negatives)-1] # remove positive
            sample_obj["candidates"] = hard_negatives + easy_negatives
            samples.append(sample_obj)

        for i in range(2):
            print("******")
            for k,v in samples[i].items():
                if k != 'candidates':
                    print(f"Data sample {i}: {k} -- {v}")
                if k == 'candidates':
                    print(f"Data sample {i}: {k} (subset) -- {v[:10]}")
            print("******\n")

        return samples


    def pad_locs(self, latlongs, truncation=True):
        latlongs = [[latlong[0]/90, latlong[1]/180] for latlong in latlongs] # normalise for question latlong
        if truncation and len(latlongs) > self.opts.max_locs:
            mask = [True] * self.opts.max_locs
            return latlongs[:self.opts.max_locs], mask
        diff = self.opts.max_locs - len(latlongs)
        mask = [True] * len(latlongs) + [False] * diff
        return latlongs + [[0.0,0.0]] * diff, mask


    def get_coordinate(self, ent):
        if 'lat_long' in self.entity_knowledge[ent]:
            latlong = self.entity_knowledge[ent]['lat_long']
        else:
            city = ent.split('_')[0]
            latlong = self.default_coordinate[city]
        return (latlong[0]/90, latlong[1]/180) # normalise for candidate latlong
        # return latlong


    def batch_process_questions(self, batch, field="question"):
        features= []
        for example in batch:
            if example['question'] in self.cached_questions:
                result = self.cached_questions[example['question']]
            else:
                texts = ((example['question'],))
                result = self.q_tokenizer(*texts, padding="max_length", max_length=self.opts.max_question_length, truncation=True)
                if self.opts.loc == 'numeric':
                    result['latlongs'], result['latlong_mask'] = self.pad_locs(example['tagged_latlongs'])
                self.cached_questions[example['question']] = result
            features.append(result)
        # convert to tensors
        questions_inputs = {}
        for k, v in features[0].items():
            questions_inputs[k] = torch.tensor([f[k] for f in features])
        return questions_inputs


    def batch_process_entities(self, batch, training=False):
        entity_latlongs = None
        entity_embeds = None
        entity_inputs = {}
        entity_loc_inputs = {}
        entity_ids = []

        if not training:
            entity_inputs['embeds'] = torch.tensor(np.array(self.entity_embeddings))
            if self.opts.loc == 'numeric':
                entity_inputs['latlongs'] = torch.tensor(np.array(self.processed_entities['latlongs']))
            return entity_inputs,{}

        # get all entity ids
        for data in batch:
            if training:
                entity_ids += [data["answer_entity_id"]]
                entity_ids += data["candidates"] #type:list
            else:
                entity_ids += data["all_answer_entity_list"]  #type:list
                entity_ids = data["candidates"] + entity_ids # put gold to the end to avoid top rank is gold if same score
        entity_indexes = [self.entity2id[e] for e in entity_ids]

        if self.opts.loc == 'numeric':
            entity_inputs['latlongs'] = torch.tensor(np.array([self.get_coordinate(i) for i in entity_ids]))

        features = [self.processed_entities[idx] for idx in entity_indexes]
        for k in features[0]:
            if k in ['input_ids','token_type_ids','attention_mask']:
                entity_inputs[k] = torch.tensor([f[k] for f in features])
            if k in ['loc_input_ids', 'loc_token_type_ids','loc_attention_mask']:
                entity_loc_inputs[k[4:]] = torch.tensor([f[k] for f in features])
        return entity_inputs, entity_loc_inputs


    def entity_tokenize_function(self, examples):
        # Tokenize the texts, use reviews only
        if self.opts.encode_entity_name:
            texts = ((examples['name'], examples['review']))
        else:
            texts = ((examples['review'],))
        result = self.e_tokenizer(*texts, padding='max_length', max_length=self.opts.max_entity_length, truncation=True)

        if self.opts.loc == 'text':
            texts = ((examples['name'],))
            result0 = self.l_tokenizer(*texts, padding='max_length', max_length=self.opts.max_entity_length, truncation=True)
            for k,v in result0.items():
                result['loc_'+k] = v
        return result


    def preprocess_all_entities(self):
        from datasets import Dataset
        df = {}
        df['id'] = self.entity_ids
        df['review'] = [' '.join(self.entity_knowledge[k]['review']) for k in df['id']]
        df['name'] = [self.entity_knowledge[k]['name'] for k in df['id']]
        if self.opts.loc == 'numeric':
            df['latlongs'] = [self.get_coordinate(k) for k in df['id']]
        entity_dataset = Dataset.from_dict(df) # create datasets class for quick process

        processed_datasets = entity_dataset.map(
            self.entity_tokenize_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        self.processed_entities = processed_datasets


    def reload_entity_embeddings(self, output_dir):
        with open(os.path.join(output_dir, self.opts.emb_file),'rb') as f:
            self.entity_embeddings = np.load(f)
        print(f"Reload pretrained entity embeddings from {output_dir}-{self.opts.emb_file}: {len(self.entity_embeddings)}")


    def prepare_forward(self, batch, training=True, device='cpu'):

        question_inputs = self.batch_process_questions(batch, field="question")
        entity_inputs, entity_loc_inputs = self.batch_process_entities(batch, training=training)

        question_inputs = {k:v.to(device) for k,v in question_inputs.items()}
        entity_inputs = {k:v.to(device) for k,v in entity_inputs.items()}
        entity_loc_inputs = {k:v.to(device) for k,v in entity_loc_inputs.items()}

        need_latlong = self.opts.loc == 'numeric'
        # seperate inputs
        q_latlongs = question_inputs['latlongs'] if need_latlong else None
        q_latlong_mask = question_inputs['latlong_mask'] if need_latlong else None
        e_latlongs = entity_inputs['latlongs'] if need_latlong else None
        e_emb = entity_inputs['embeds'] if not training else None
        entity_loc_inputs = entity_loc_inputs if (training and self.opts.loc=='text') else None
        question_inputs.pop('latlongs',None)
        question_inputs.pop('latlong_mask',None)
        entity_inputs.pop('latlongs',None)
        entity_inputs.pop('embeds',None)

        batch = ({"question_inputs": question_inputs,
                  "question_latlongs": q_latlongs,
                  "question_latlong_mask": q_latlong_mask,
                  "entity_inputs":entity_inputs,
                  "entity_loc_inputs":entity_loc_inputs,
                  "entity_latlongs":e_latlongs,
                  "entity_vectors":e_emb,
                 })
        return batch

