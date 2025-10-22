import torch
import random
import os
import json

import pandas as pd
import numpy as np

from tqdm import tqdm
from dataset import preprocess_datasets, DatasetForGeneration, prepare_hellaswag_data, Text2Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pydantic import BaseModel
from omegaconf import OmegaConf
from dataset import *
from plt_model import LitModel

class Config(BaseModel):
    tag_model: str = 'facebook/opt-125m'
    stage: int = 1
    total_fractions: int = 2
    load_name: str = None
    block_size: int = 64
    batch_size: int = 128
    seed: int = 0
    pretrained: bool = False
    num_workers: int = 127
    generate: str = None
    load_generate: str = None
    generated_length: int = 64
    generation_context_length: int = 512
    use_consistency_block: bool = False
    task: str = "wikitext"
    class Config:
        extra = "forbid"

@torch.no_grad()
def generate_dataset(plt_model, train_dataset, args, tokenizer, task):
    if task == "hellaswag":
        generated_completions = []
        # Generating the new responses
        for batch in tqdm(train_dataset):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            model = plt_model.model.eval()
            model = model.cuda()

            outputs = model.generate(input_ids,
                                     max_new_tokens=512,
                                     attention_mask=attention_mask,
                                     pad_token_id=model.config.eos_token_id,)
            input_length = input_ids.shape[1]
            preds = outputs[:, input_length:]

            for i in range(preds.shape[0]):
                generated_completions.append(tokenizer.decode(preds[i, :], skip_special_tokens=True))

        prev_dataset = load_dataset('hellaswag')
        prev_dataset.pop("test")
        prev_dataset.pop("validation")

        for i in range(len(generated_completions)):
            # Replacing the generated text in the old dataset with the correct answer
            correct_answer = int(prev_dataset["train"][i]['label'])
            prev_dataset["train"][i]['endings'][correct_answer] = generated_completions[i]
        # Saving the new dataset
        # Create folder if it doesn't exists
        os.makedirs(os.path.dirname(args.load_generate), exist_ok=True)
        # Save the dataset
        prev_dataset.to_json(args.load_generate + f"stage{args.stage}.json", orient="records", lines=True, indent=4)

    if task == "wikitext" or task == "imagination":
        dataset_for_generation = DatasetForGeneration(train_dataset, context_length=args.generation_context_length)
        train_dataloader = DataLoader(dataset_for_generation,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      )

        model = plt_model.model.eval()
        generated_batches = ""
        if args.use_consistency_block:
            consistency_block = torch.tensor([]).cuda()
        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].cuda()
            if args.use_consistency_block:
                try:
                    input_ids = torch.concat((input_ids[:, -args.generated_length: ], consistency_block), dim=1).long()
                except:
                    pass
            attention_mask = batch["attention_mask"].cuda()
            model = model.cuda()

            batch_size, _ = input_ids.shape
            outputs = model.generate(input_ids,
                                     num_beams=5,
                                     max_new_tokens=args.generated_length,
                                     attention_mask=attention_mask,
                                     min_new_tokens=args.generated_length,
                                     repetition_penalty=3.0,
                                     pad_token_id=model.config.eos_token_id,)
            input_length = input_ids.shape[1]
            # Skipping the input
            preds = outputs[:, input_length:]
            batch["input_ids"] = preds.cpu().detach()
            batch["labels"]    = preds.cpu().detach()
            batch["attention_mask"] = attention_mask.cpu().detach()

            # Clean up memory
            del input_ids
            del attention_mask
            if args.use_consistency_block:
                consistency_block = preds

            for j in range(batch_size):
                generated_batches += (tokenizer.decode(batch["input_ids"][j, :]))

        file_name = args.generate
        print(type(generated_batches))
        # Create foler if it doesn't exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            f.write(generated_batches)
            os.system(f"chmod -R 777 {f}")
            f.close()
        print(f"{file_name} generated and saved!")
        return generated_batches

def main(args=None):
    if args is  None:
        conf_dict = OmegaConf.from_cli()
    else :
        conf_dict = args
    args = Config(**conf_dict)
    print(args)
    tag_model = args.tag_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.pretrained:
        if "gemma" in args.tag_model:
            model = AutoModelForCausalLM.from_pretrained(args.tag_model, cache_dir='./model_cache_dir', attn_implementation='eager')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.tag_model, cache_dir='./model_cache_dir')
    else:
        config = AutoConfig.from_pretrained(tag_model)
        model = AutoModelForCausalLM.from_config(config=config)

    tokenizer = AutoTokenizer.from_pretrained(tag_model,
                                              cache_dir='./model_cache_dir',
                                              return_dict=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print(f"Loaded tokenizer from HuggingFace")

    if args.task == "wikitext":
        raw_dataset = prepare_data()
        dataset = preprocess_datasets(raw_dataset, tokenizer, block_size=args.block_size, task=args.task)
        training_dataset = MixedDataset(generated_dataset_folder=args.load_generate,
                                        original_dataset=dataset,
                                        experiment_stage=args.stage,
                                        fractions=args.total_fractions,
                                        block_size=args.block_size,
                                        partition='train',
                                        tokenizer=tokenizer,
                                        task=args.task,
                                        )
    if args.task == "imagination":
        dataset_file = "imagination_dataset.txt"
        raw_dataset = load_dataset("text", data_files={"train": [dataset_file]})
        dataset = preprocess_datasets(raw_dataset, tokenizer, block_size=args.block_size, task=args.task)
        training_dataset = MixedDataset(generated_dataset_folder=args.load_generate,
                                        original_dataset=dataset,
                                        experiment_stage=args.stage,
                                        fractions=args.total_fractions,
                                        block_size=args.block_size,
                                        partition='train',
                                        tokenizer=tokenizer,
                                        task=args.task,
                                        )
    if args.task == "hellaswag":
        generated_dataset_file = args.load_generate + f"stage{args.stage - 1}.json"
        # First stage
        if args.stage == 0:
            generated_dataset = prepare_hellaswag_data(for_generation=True)
        else:
            # Stage 1 and above
            prev_dataset = json.load(open(generated_dataset_file, 'r'))
            generated_dataset = prepare_hellaswag_data(prev_dataset, for_generation=True)
        
        generated_dataset = preprocess_datasets(generated_dataset, tokenizer, block_size=args.block_size, task=args.task)
        generated_dataset = Text2Dataset(generated_dataset, partition="train", tokenizer=tokenizer)
        training_dataset = DataLoader(generated_dataset,
                                      batch_size=256,
                                      num_workers=args.num_workers)
        print(training_dataset)
    
    # Loading the model
    if args.load_name is not None:
        plt_model = LitModel.load_from_checkpoint(args.load_name, map_location="cpu")
    else:
        plt_model = LitModel(model)

    generate_dataset(plt_model, training_dataset, args, tokenizer=tokenizer, task=args.task)
    return

if __name__ == '__main__':
    main()