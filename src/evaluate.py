import json
import torch
import pickle
import random

import numpy as np

from tqdm import tqdm
from dataset import preprocess_datasets, DatasetForGeneration
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pydantic import BaseModel
from omegaconf import OmegaConf
from dataset import *
from plt_model import LitModel
from scipy.stats import chi2
from evaluate_utils import *

class Config(BaseModel):
    tag_model: str = 'facebook/opt-125m'
    load_name: str = None
    save_name: str = None
    seed: int = 0
    pretrained: bool = False
    version_name: str = None
    generate: str = None
    load_generate: str = None
    original_dataset_fraction: float = 0.0
    accumulate_grad_batches: int = 1
    generated_length: int = 64
    evaluate_KT: str = None
    stage: int = 1
    total_fractions: int = 2
    eval_fractions: bool = True
    delta: int = 0
    class Config:
        extra = "forbid"

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
        if "gemma" in tag_model:
            model = AutoModelForCausalLM.from_pretrained(tag_model, cache_dir='./model_cache_dir', attn_implementation='eager')
        else:
            model = AutoModelForCausalLM.from_pretrained(tag_model, cache_dir='./model_cache_dir')
    else:
        config = AutoConfig.from_pretrained(tag_model)
        model = AutoModelForCausalLM.from_config(config=config)

    tokenizer = AutoTokenizer.from_pretrained(tag_model,
                                              cache_dir='./model_cache_dir',
                                              return_dict=True
                                              )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded tokenizer from HuggingFace")
    if args.load_name is not None:
        plt_model = LitModel.load_from_checkpoint(args.load_name, map_location="cpu")
    else:
        plt_model = LitModel(model)

    dataset_dict = json.load(open(args.evaluate_KT))
    print(f'Loaded {len(dataset_dict)} questions from {args.evaluate_KT}')


    fractioned_exam_taker(model=plt_model.model, 
                         tokenizer=tokenizer, 
                         exam=dataset_dict, 
                         stage=args.stage, 
                         fractions=args.total_fractions,
                         delta=args.delta,
    )

if __name__ == '__main__':
    main()