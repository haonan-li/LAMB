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

You can download the dataset we used in our experiment from [Google drive](https://drive.google.com/file/d/19rConnPIkZMdPDEHqkEgt7HpQRgqFwH5/view?usp=sharing)

### Training

To train the LAMB model, enter the `src` directory, and run `LAMB_Exec.py`.

This example code below trains the model with negative size of 8 and batch size of 8, it can be trained on a single tesla V100 16GB GPU.
To reproduce the best results reported in the paper, use the 32GB or 40 GB GPU and the corresponding experimental setup.

```bash
python LAMB_Exec.py
    --transformer_model distilbert-base-uncased \
    --data_dir ../data \
    --encode_entity_name \
    --location \
    --distance \
    --lr 2e-5 \
    --loss nll \ 
    --score_method dot 
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --samples_per_qa 9 \
    --hard_negatives_per_qa 8 \
    --n_cluster_reviews 5 \
    --train_file train.json \
    --test_file test.json \
    --num_train_epochs 20 \
    --s2_after 12 \
    --max_locations 5 \
    --dist_weight 0.1 \
    --batch_size_save_entity 256 \
    --output_dir ../output \
```
