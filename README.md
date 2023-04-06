# LAMB
LAMB: Location-Aware Modular Bi-encoder for Tourism Question Answering

## Dependencies

* python>=3.6
* torch>=1.11.0
* trnasformers>=4.19.2
* datasets>=2.2.2
* huggingface-hub>=0.6.0

## Quick Start for Training and Inference

### Download the data

You can download the model parameters and dataset from [Fake link]().
Note: This repo is currently anonymoused, we will release link on acceptance.

### Training

To train the LAMB model, enter the `src` directory, and run `LAMB_Exec.py`.

This example code below trains the model with negative size of 8 and batch size of 8, it can be trained on a single tesla V100 16GB GPU.
To reproduce the best results reported in the paper, use the 32GB or 40 GB GPU and the corresponding experimental setup.

```bash
python LAMB_Exec.py
    --q_encoder distilbert-base-uncased \
    --e_encoder distilbert-base-uncased \
    --l_encoder ../data/loc_module/loc_2layer.pth \
    --data_dir ../data \
    --location text \
    --lr 2e-5 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --samples_per_qa 8 \
    --hard_negatives_per_qa 5 \
    --train_file train.json \
    --test_file test.json \
    --num_train_epochs 10 \
    --s2_after 5 \
    --output_dir ../output \
    --fp16
```
