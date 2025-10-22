from generate import main as generate_func
from train import main as train_func
import os

model_parameter = {"facebook/opt-125m": "125m",
                   "facebook/opt-350m": "350b",
                   "facebook/opt-1.3b": "1.3b",
                   "meta-llama/Llama-3.2-1B": "llama1b",
                   "microsoft/phi-2": "phi-2",
                   }

train_args = {
    "tag_model": "meta-llama/Llama-3.2-1B",
    "pretrained": True,
    "load_name": "/scr/echollm/checkpoints/baseline/frac3/meta-llama/Llama-3.2-1B/M0_f3-llama1b-batch16-lr2e-05-seed8/last.ckpt",
    "load_generate": "/scr/echollm/generated_datasets/llama1b/baseline/frac3/batch16-lr2e-05-seed8/",
    "batch_size": 16,
    "num_devices": 2,
    "accumulate_grad_batches": 1,
    "num_workers": 127,
    "block_size": 512,
    "max_epochs": 10,
    "strategy": "ddp",
    "seed": 9,
    "learning_rate": 2e-5,
    "optimizer": "adam",
    "blimp_eval": ['adjunct_island', 'causative'],
    "evaluate_KT": "kt_dataset_gpt_mini.json",
    "stage": 1,
    "dataset_fractions": 3,
}

train_args['save_name'] = (
    f"/scr/echollm/checkpoints/baseline/frac{train_args['dataset_fractions']}/{train_args['tag_model']}/"
    f"M{train_args['stage']}_f{train_args['dataset_fractions']}-"
    f"{model_parameter[train_args['tag_model']]}-"
    f"batch{train_args['batch_size']}-"
    f"lr{train_args['learning_rate']}-"
    f"seed{train_args['seed']}"
)
train_args['version_name'] = (
    f"Soheil/M{train_args['stage']}/{train_args['dataset_fractions']}-"
    f"{model_parameter[train_args['tag_model']]}-"
    f"{train_args['batch_size']}-"
    f"lr{train_args['learning_rate']}-"
    f"seed{train_args['seed']}"
)

train_func(train_args) 

os.system(f"chmod -R 777 {train_args['save_name']}")