import re
import math
import string
import os
import torch
import pytorch_lightning as pl
import pickle
import json

from datasets import Dataset as HFDataset

from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
from itertools import chain
from transformers import default_data_collator


def prepare_data(path='./data/wikitext2'):
    if (path is not None) and (not os.path.isdir(path)):
        print("Downloading and processing dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        dataset.save_to_disk(path)
    else:
        print("Dataset already downloaded and processed")
        dataset = load_from_disk(path)
    dataset.pop("test")
    # dataset.pop("validation")
    return dataset


def prepare_hellaswag_data(path='./data/hellaswag', dataset=None, for_generation=False):
    # Prepares the HellaSwag dataset for training and evaluation
    # Converting from Arrow dataset or Json to HFDataset with text and labels_gt columns
    print("Downloading and processing dataset...")
    if dataset is None:
        dataset = load_dataset('hellaswag')
        dataset.pop("test")
    prepared_dataset = {"train": {}, "validation": {}}

    if for_generation:
        print("Tokenizing only the context")
    else:
        print("Tokenizing the context and the correct ending")

    for partition in dataset.keys():
        texts = []
        labels = []
        for i in range(len(dataset[partition])):
            text = [f"{ctx}{ending}" for ctx, ending in zip([dataset[partition][i]["ctx"]] * 4, dataset[partition][i]["endings"])]
            label = int(dataset[partition][i]["label"])
            if partition == "train":
                # Only the correct label is used for training
                if for_generation:
                    # For generation, we need to generate the correct answer
                    texts.append(dataset[partition][i]["ctx"])
                else:
                    texts.append(text[label])
                labels.append(label)
            else:
                # Validation set uses all the labels
                for j in range(len(text)):
                    texts.append(text[j])
                    labels.append(label)

        # The validation set is too big (40,000), so we need to limit it
        
        texts = texts[:128]
        labels = labels[:128]

        partition_dataset = HFDataset.from_dict({"text": texts, "labels_gt": labels})
        prepared_dataset[partition] = partition_dataset

    dataset = DatasetDict(prepared_dataset)
    return dataset


def preprocess_datasets(raw_dataset,
                        tokenizer,
                        block_size=64,
                        overwrite_cache=False, 
                        preprocessing_num_workers=4,
                        task="wikitext",
                       ):
    # Tokenizes the dataset and/or groups the texts into chunks of block_size (for wikitext)
    column_names = raw_dataset['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if block_size is None:
        block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        if task == "wikitext" or "imagination":
            tokenized = tokenizer(examples[text_column_name])
        if task == "hellaswag":
            tokenized = tokenizer(examples[text_column_name],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=512,
                                 )
            tokenized["labels_gt"] = examples["labels_gt"]
        return tokenized

    tokenized_datasets = raw_dataset.map(tokenize_function,
                                         batched=True,
                                         num_proc=preprocessing_num_workers,
                                         remove_columns=column_names,
                                         load_from_cache_file=not overwrite_cache,
                                         desc="Running tokenizer on dataset",
                                         keep_in_memory=True,
                                        )
    print(tokenized_datasets)
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def hellaswag_texts(examples):
        result = examples
        result["labels"] = result["input_ids"].copy()
        return result

    if task == "wikitext" or task == "imagination":
        # Main dataset is already tokenized, so we just need to group the texts
        # into chunks of block_size
        print("Grouping texts in chunks of", block_size)
        dataset = tokenized_datasets.map(group_texts,
                                         batched=True,
                                         num_proc=preprocessing_num_workers,
                                         load_from_cache_file=not overwrite_cache,
                                         desc=f"(Preprocess datasets) Grouping texts in chunks of {block_size} for {task}",
                                         keep_in_memory=True
                                        )
    if task == "hellaswag":
        dataset = tokenized_datasets.map(hellaswag_texts,
                                         batched=True,
                                         num_proc=preprocessing_num_workers,
                                         load_from_cache_file=not overwrite_cache,
                                         desc=f"(Preprocess datasets) Adding labels and ground-truth choices for the validaiton set",
                                         keep_in_memory=True
                                        )
        
    return dataset


def prepare_blimp_data(path='./data/blimp', dataset_name=['adjunct_island', 'causative']):
    path = path + '_'.join(dataset_name)
    if os.path.exists(path):
        return load_from_disk(path)
    print("Downloading and processing dataset...")
    dataset = []
    for name in dataset_name:
        dataset.append(load_dataset('nyu-mll/blimp', name)['train'])
    dataset = concatenate_datasets(dataset)
    dataset.save_to_disk(path)
    return dataset


class Text2Dataset(Dataset):
    def __init__(self, 
                 dataset,
                 partition,
                 tokenizer=None, 
                 max_token_count=512):
        self.setup_tokenizer(tokenizer, max_token_count)
        self.dataset = dataset[partition]

    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        data_row = self.dataset[index]
        if 'labels_gt' in data_row:
            return dict(input_ids=torch.tensor(data_row['input_ids']),
                        attention_mask=torch.tensor(data_row['attention_mask']),
                        labels=torch.tensor(data_row['labels']),
                        labels_gt=torch.tensor(data_row['labels_gt']),
                        )
        return dict(input_ids=torch.tensor(data_row['input_ids']),
                    attention_mask=torch.tensor(data_row['attention_mask']),
                    labels=torch.tensor(data_row['labels']),
                    labels_gt=torch.tensor(data_row['labels']),
                    )

class GeneratedDataset_V2(Dataset):
    def __init__(self,
                 file,
                 block_size=64,
                 partition='train',
                 tokenizer=None,
                 max_token_count=512):
        with open(file, 'rb') as f:
            train_dataset = pickle.load(f)
        train_dataset = HFDataset.from_dict({"input_ids": [d["input_ids"] for d in train_dataset],
                                             "attention_mask": [d["attention_mask"] for d in train_dataset]})
        train_dataset = DatasetDict({"train": train_dataset})
        # concatenated_ids = torch.stack([train_dataset[i]["input_ids"] for i in range(len(train_dataset))], dim=0).reshape(-1)
        # concatenated_attention_mask = torch.stack([train_dataset[i]["attention_mask"] for i in range(len(train_dataset))], dim=0).reshape(-1)
        # concatenated_dataset = {
        #     "input_ids": concatenated_ids,
        #     "attention_mask": concatenated_attention_mask
        # }
        def group_texts(examples):
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])

            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size]
                    for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        dataset_ = train_dataset.map(group_texts,
                                     batched=True,
                                     num_proc=1,
                                     load_from_cache_file=False,
                                     desc=f"(Generated datasets) Grouping texts in chunks of {block_size}",
                                     keep_in_memory=True
                                    )
        self.dataset = dataset_['train']

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        data_row = self.dataset[index]
        return dict(input_ids=torch.tensor(data_row['input_ids']),
                    attention_mask=torch.tensor(data_row['attention_mask']),
                    labels=torch.tensor(data_row['labels']))


class MixedDataset(Dataset):
    def __init__(self,
                 generated_dataset_folder,
                 original_dataset,
                 experiment_stage,
                 fractions,
                 block_size=64,
                 partition='train',
                 tokenizer=None,
                 task='wikitext',
                 scale=0,
                 ):
        self.task = task
        if task == "hellaswag":
            assert not ((experiment_stage > 0) and (generated_dataset_folder is None)), "You need to pass the generated_dataset_folder if you're not at the first stage"        
            generated_dataset_file = generated_dataset_folder + f"stage{experiment_stage - 1}.json"
            try:
                generated_dataset = json.load(open(generated_dataset_file, 'r'))
                generated_dataset = prepare_hellaswag_data(path=generated_dataset_folder, dataset=generated_dataset)
                generated_dataset = generated_dataset["train"]
                generated_dataset = preprocess_datasets(generated_dataset,
                                                        tokenizer=tokenizer,
                                                        task=task,)
            except FileNotFoundError:
                print(f"File {generated_dataset_file} not found")
                generated_dataset = HFDataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
            # We will only generate the correct answer of train dataset again
            self.dataset = generated_dataset
        if task == "wikitext" or task == "imagination":
            assert not ((experiment_stage > 0) and (generated_dataset_folder is None)), "You need to pass the generated_dataset_folder if you're not at the first stage"        
            if generated_dataset_folder is not None:
                generated_dataset_file = generated_dataset_folder + f"stage{experiment_stage - 1}.txt"
                try:
                    generated_dataset_uptostage = load_dataset("text", data_files={"train": [generated_dataset_file]})
                    train_dataset = preprocess_datasets(generated_dataset_uptostage,
                                                        tokenizer=tokenizer,
                                                        task=task,)["train"] 
                except FileNotFoundError:
                    print("^" * 25, f"File {generated_dataset_file} not found", "^" * 25)
                    # First stage or file not found
                    train_dataset = HFDataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
            else:
                train_dataset = HFDataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})

            print("generated_dataset num_rows", train_dataset.num_rows)
            print("original_dataset num_rows", original_dataset[partition].num_rows)

            # Scale = 2 is for 50% synthetic
            fraction_size = original_dataset[partition].num_rows // fractions
            if scale > 0:
                if experiment_stage == 0:
                    fractioned_dataset = HFDataset.from_dict(original_dataset[partition][experiment_stage * fraction_size: (experiment_stage + 1) * fraction_size])
                else:
                    fractioned_dataset = HFDataset.from_dict(original_dataset[partition][experiment_stage * fraction_size: scale * fraction_size])
            else:
                fractioned_dataset = HFDataset.from_dict(original_dataset[partition][experiment_stage * fraction_size: (experiment_stage + 1) * fraction_size])

            # Double checking that the split after tokenizing and grouping is accurate enough to be used
            # This is also relatively accurate for the evaluation to just split the questions into fractions
            # half of dataset is ~18k lines which is ~15k of the questions
            # print(tokenizer.decode(fractioned_dataset["input_ids"][-2]))
            # print(tokenizer.decode(fractioned_dataset["input_ids"][-1]))
            
            print("fractioned_dataset num_rows", fractioned_dataset.num_rows)
            print(train_dataset)
            train_dataset = concatenate_datasets((train_dataset, fractioned_dataset))
            print("train_dataset num_rows", train_dataset.num_rows)
            print(train_dataset)
            train_dataset = DatasetDict({"train": train_dataset})

            def group_texts(examples):
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])

                if total_length >= block_size:
                    # Discarding the remainder
                    total_length = (total_length // block_size) * block_size
                result = {k: [t[i: i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()}
                result["labels"] = result["input_ids"].copy()
                return result

            dataset_ = train_dataset.map(group_texts,
                                         batched=True,
                                         num_proc=1,
                                         load_from_cache_file=False,
                                         desc=f"(MixedDataset) Grouping texts in chunks of {block_size}",
                                         keep_in_memory=True)
            self.dataset = dataset_['train']

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        data_row = self.dataset[index]
        if 'labels_gt' in data_row:
            return dict(input_ids=torch.tensor(data_row['input_ids']),
                        attention_mask=torch.tensor(data_row['attention_mask']),
                        labels=torch.tensor(data_row['labels']),
                        labels_gt=torch.tensor(data_row['labels_gt']),
                        )
        return dict(input_ids=torch.tensor(data_row['input_ids']),
                    attention_mask=torch.tensor(data_row['attention_mask']),
                    labels=torch.tensor(data_row['labels']),
                    labels_gt=torch.tensor(data_row['labels']),
                    )


class DatasetForGeneration(Dataset):
    def __init__(self, dataset, context_length=512,):
        self.dataset = dataset
        block_size = len(dataset[0]['input_ids'])
        self.blocks_before = context_length // block_size

    def __len__(self):
        return self.dataset.__len__() - self.blocks_before

    def __getitem__(self, index):
        upper_index = min(index + self.blocks_before, len(self.dataset))
        data = self.dataset[index:upper_index]
        input_ids = data['input_ids'].view(-1)
        attention_mask = data['attention_mask'].view(-1)
        labels = data['labels'].view(-1)
        return dict(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,)