import os,sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch.nn as nn
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


parser = argparse.ArgumentParser(description='Lamb Loc Module')
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--n_loc_layers", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_train_epochs", type=int, default=12)
parser.add_argument("--l_encoder", type=str, default='distilbert-base-uncased')
parser.add_argument("--knowledge_file", type=str,default="TourQue_Knowledge_Sel.json")
parser.add_argument('--data_dir', type=str, default="../data/")
parser.add_argument("--max_loc_length", type=int, default=64)
args = parser.parse_args(sys.argv[1:])



# Data
with open(os.path.join(args.data_dir, args.knowledge_file)) as f:
    data = pd.read_json(f, orient='index')
data = data.dropna() # remove the lines without latlong

# Create a PyTorch dataset
class PlacesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, latlongs):
        self.encodings = encodings
        self.latlong = latlongs

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['latlong'] = torch.tensor(self.latlong[idx])
        return item

    def __len__(self):
        return len(self.latlong)

# Model
model = AutoModel.from_pretrained(args.l_encoder)
model.transformer.layer = model.transformer.layer[:args.n_loc_layers] # keep only n_layers
tokenizer = AutoTokenizer.from_pretrained(args.l_encoder)

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

train_encodings = tokenizer(train_data['name'].tolist(), padding=True, truncation=True, max_length=args.max_loc_length)
test_encodings = tokenizer(test_data['name'].tolist(), padding=True, truncation=True, max_length=args.max_loc_length)

# Create the PyTorch datasets and data loaders
train_dataset = PlacesDataset(train_encodings, train_data['lat_long'].tolist())
test_dataset = PlacesDataset(test_encodings, test_data['lat_long'].tolist())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Train
# Define hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the optimizer and the learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
criterion = torch.nn.MSELoss()

# Model
import math
import torch
import torch.nn as nn
import numpy as np

class PlaceDistanceLoss(nn.Module):
    def __init__(self, margin):
        super(PlaceDistanceLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, coords, device):
        # embeddings is the tensor of fixed-length representations of the place names
        # coords is the tensor of geocoordinates

        # Compute pairwise distances between all pairs of places
        # Convert latitude and longitude to radians
        coordinates_rad = torch.deg2rad(coords)

        # Compute pairwise differences in latitude and longitude
        dlat = coordinates_rad[:, None, 0] - coordinates_rad[None, :, 0]
        dlon = coordinates_rad[:, None, 1] - coordinates_rad[None, :, 1]

        # Compute great-circle distance using Haversine formula
        a = torch.sin(dlat / 2) ** 2 + torch.cos(coordinates_rad[:, 0])[:, None] * torch.cos(coordinates_rad[None, :, 0]) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distances = c / math.pi  # Earth radius = 6371 km

        # distance is a tensor of shape (num_points, num_points) containing the true pairwise distances

        # Compute pairwise cosine similarities between all pairs of fixed-length representations
        similarities = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

        # Convert distances to similarities using a Gaussian kernel
        sigma = distances.std()
        similarities_gt = torch.exp(-distances ** 2 / (2 * sigma ** 2))

        # Compute the triplet loss
        margin = self.margin
        N = embeddings.size(0)
        loss = torch.tensor(0.0).to(device)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                for k in range(j):
                    if k == i or k == j:
                        continue

                    a_sim = similarities[i, j]
                    b_sim = similarities[i, k]
                    a_sim_gt = similarities_gt[i, j]
                    b_sim_gt = similarities_gt[i, k]

                    flag = a_sim_gt - b_sim_gt
                    # print(pos_sim, neg_sim, pos_sim_gt, neg_sim_gt)
                    # Compute the triplet loss
                    if flag < 0:
                        triplet_loss = torch.max(a_sim - b_sim - flag, torch.tensor(0.0).to(device))
                    else:
                        triplet_loss = torch.max(b_sim - a_sim + flag, torch.tensor(0.0).to(device))
                    # Add the triplet loss to the total loss
                    loss += triplet_loss # + triplet_loss_gt

        # Normalize the loss by the number of triplets
        num_triplets = N * (N - 1) * (N - 2) / 2
        loss /= num_triplets

        return loss
criterion = PlaceDistanceLoss(margin=2)

# Train
# Define the training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        latlong = batch['latlong'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        rep = outputs.last_hidden_state[:,0,:]
        loss = criterion(rep, latlong.float(), device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Define the evaluation loop
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            latlong = batch['latlong'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            rep = outputs.last_hidden_state[:,0,:]
            loss = criterion(rep, latlong.float(), device)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Train the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
for epoch in range(args.num_train_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss = evaluate(model, test_loader, criterion, device)
    # Save to data_dir
    model.save_pretrained(os.path.join(args.data_dir, 'loc_module', f'loc_{args.n_layers}layer.pth'))
