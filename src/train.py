import argparse
import torch
import random
import pickle
import os
import copy
import json

import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM, default_data_collator
from pytorch_lightning.callbacks import ModelCheckpoint
from plt_model import LitModel
from dataset import Text2Dataset, prepare_data, preprocess_datasets, MixedDataset, prepare_hellaswag_data
from datasets import Dataset as HFDataset, load_dataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pydantic import BaseModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
class Config(BaseModel):
    tag_model: str = 'facebook/opt-125m'
    loss_type: str = 'baseline'
    load_name: str = None
    save_name: str = None
    saveperplexities: str = None
    eval_only: bool = False
    evalgen_only: bool = False
    optimizer: str = 'adam'
    focal_gamma: float = None
    focal_alpha: float = None
    learning_rate: float = 2e-5
    max_epochs: int = 5
    block_size: int = 64
    batch_size: int = 128
    debug: bool = False
    seed: int = 0
    num_workers: int = 127
    num_devices: int = 1
    accelerator: str = "auto"
    strategy: str = 'ddp'
    pretrained: bool = False
    version_name: str = None
    generate: str = None
    load_generate: str = None
    original_dataset_fraction: float = 0.0
    accumulate_grad_batches: int = 1
    generated_length: int = 64
    evaluate_KT: str = "kt_dataset_full.json"
    blimp_eval: list = ['adjunct_island', 'causative']
    dataset_fractions: int = 1
    stage: int = 1
    access_token: str = "/home/pedram/echo-llm/huggingface_token.txt"
    test_mode: bool = False
    tags: list = []
    task: str = "wikitext"
    scale: int = 0
    
    class Config:
        extra = "forbid"

def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model


def load_model(load_path, plt_model):
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            raise ValueError("Invalid checkpoint file. must end with .ckpt")
        plt_model = plt_model_load(plt_model, checkpoint)
        print(f"Loaded model from {checkpoint}")
    return plt_model


def main(args=None):

    if args is None:
        conf_dict = OmegaConf.from_cli()
    else :
        conf_dict = args
    args = Config(**conf_dict)

    if args.load_name == "":
        args.load_name = None
    if args.load_generate == "":
        args.load_generate = None

    args.batch_size = args.batch_size // args.num_devices // args.accumulate_grad_batches
    print(args)

    if args.test_mode == True:
        tags = ["test"]
    else:
        tags = args.tags + [args.task]
        if args.scale > 0:
            tags.append(f"scale{args.scale}")
            tags.append("short-term")

    # Seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pl.seed_everything(args.seed)

    
    if args.access_token is not None or not os.path.exists(args.access_token):
        with open(args.access_token, "r") as f:
            token = f.read().strip()
    else:
        token = None

    if args.pretrained:
        if "gemma" in args.tag_model:
            model = AutoModelForCausalLM.from_pretrained(args.tag_model, cache_dir='./model_cache_dir', token=token, attn_implementation='eager')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.tag_model, cache_dir='./model_cache_dir', token=token)
    else:
        config = AutoConfig.from_pretrained(args.tag_model)
        model = AutoModelForCausalLM.from_config(config=config)
    print(f"Loaded model from HuggingFace")

    tokenizer = AutoTokenizer.from_pretrained(args.tag_model,
                                              cache_dir='./model_cache_dir',
                                              return_dict=True,
                                              token=token,
                                             )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Loaded tokenizer from HuggingFace")

    if args.task == "wikitext":
        raw_dataset = prepare_data()
    if args.task == "hellaswag":
        if args.stage == 0:
            raw_dataset = prepare_hellaswag_data()
        else:
            raw_dataset = HFDataset.from_dict(json.load(open(args.load_generate + f"stage{args.stage - 1}.json")))
    if args.task == "imagination":
        dataset_file = "imagination_dataset.txt"
        raw_dataset = load_dataset("text", data_files={"train": [dataset_file]})
        val_dataset = prepare_data()
        print("Raw dataset is", raw_dataset)
        print("Val dataset is", val_dataset)
        raw_dataset["validation"] = val_dataset["validation"]


    dataset = preprocess_datasets(raw_dataset, tokenizer, block_size=args.block_size, task=args.task)

    print(f"Loaded dataset from {args.load_generate}")
    val_dataset = copy.deepcopy(Text2Dataset(dataset=dataset, partition='validation', tokenizer=tokenizer))
    # Only keeping the first 1000 samples for validation

    if args.task == "wikitext" or args.task == "imagination":
        train_dataset = MixedDataset(generated_dataset_folder=args.load_generate,
                                     original_dataset=dataset, 
                                     experiment_stage=args.stage,
                                     fractions=args.dataset_fractions,
                                     block_size=args.block_size,
                                     partition='train',
                                     tokenizer=tokenizer,
                                     task=args.task,
                                     scale=args.scale,
                                    )
    else:
        train_dataset = Text2Dataset(dataset=dataset, partition='train', tokenizer=tokenizer)
    # if args.evalgen_only:
    #     test_dataset = train_dataset
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    if args.task == "wikitext" or args.task == "imagination":
        metric = "KT_total/accuracy"
    if args.task == "hellaswag":
        metric = "Hellaswag/accuracy"

    checkpoint_callback=ModelCheckpoint(
        save_top_k=1,
        monitor=metric,
        mode="max",
        filename="best",
        dirpath=args.save_name,
        save_last=True,
        enable_version_counter=True,
    )

    clip_dataset = None
    if args.loss_type.startswith("dynamic"):
        # Randomly sample 1024 indices
        clip_dataset = []
        dataset_size = 2048
        for batch in train_dataloader:
            if len(clip_dataset) > dataset_size / args.batch_size:
                break
            clip_dataset.append(batch)

    print("!" * 60)
    print(f"Loading model from {args.load_name}")

    if args.load_name is not None:
        plt_model = LitModel.load_from_checkpoint(args.load_name,
                                                  map_location="cpu",
                                                  learning_rate=args.learning_rate,
                                                  epochs=args.max_epochs, 
                                                  focal_gamma=args.focal_gamma, 
                                                  focal_alpha=args.focal_alpha,
                                                  evaluate_KT=args.evaluate_KT, 
                                                  evaluate_hellaswag=raw_dataset["validation"],
                                                  tokenizer=tokenizer,
                                                  optimizer=args.optimizer,
                                                  blimp_eval=args.blimp_eval,
                                                  fractions=args.dataset_fractions,
                                                  stage=args.stage,
                                                  loss_type=args.loss_type,
                                                  test_mode=args.test_mode,
                                                  clip_dataset=clip_dataset,
                                                  task=args.task,
                                                 )
    else:
        plt_model = LitModel(model,
                            learning_rate=args.learning_rate,
                            epochs=args.max_epochs, 
                            focal_gamma=args.focal_gamma, 
                            focal_alpha=args.focal_alpha,
                            evaluate_KT=args.evaluate_KT,
                            evaluate_hellaswag=raw_dataset["validation"],
                            tokenizer=tokenizer,
                            optimizer=args.optimizer,
                            blimp_eval=args.blimp_eval,
                            fractions=args.dataset_fractions,
                            stage=args.stage,
                            loss_type=args.loss_type,
                            test_mode=args.test_mode,
                            clip_dataset=clip_dataset,
                            task=args.task,
                        )
    
    logger = WandbLogger(name=args.version_name, project="echo_llm", log_model=False, config=args.model_dump(), tags=tags)

    #testm = MetricTracker()#MetricTracker(MetricCollection([CatMetric()]))

    
    trainer=pl.Trainer(max_epochs=args.max_epochs,
                       devices=args.num_devices,
                       accelerator=args.accelerator,
                       strategy=args.strategy,
                       fast_dev_run=args.debug,
                       accumulate_grad_batches=args.accumulate_grad_batches,
                       callbacks=[checkpoint_callback],
                       logger=logger,
                      )

    # plt_model = plt_model.cuda()
    if not args.eval_only:
        print("Validation before training")
        trainer.validate(plt_model, dataloaders=val_dataloader)
        print("Training")
        trainer.fit(plt_model, 
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    )
    
    if args.saveperplexities:
        plt_model.tosave=True

if __name__ == '__main__':
    main()
