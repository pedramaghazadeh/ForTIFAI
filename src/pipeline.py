import argparse
import os

from generate import main as generate_func
from train import main as train_func
from aws import connect_to_s3, download_file, upload_file, upload_folder
from datetime import datetime

model_parameter = {"facebook/opt-125m": "125m",
                   "facebook/opt-350m": "350b",
                   "facebook/opt-1.3b": "1.3b",
                   "meta-llama/Llama-3.2-1B": "llama1b",
                   "microsoft/phi-2": "phi-2",
                   "google/gemma-2-2b-it": "gemma2b",
                   "google/gemma-3-1b-pt": "gemma3b",
                   "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek1.5",
                   }

def run_stage(args):
    stage = args.stage

    if args.server == "aws":
        prefix_path = "/opt/dlami/nvme"
    else:
        prefix_path = "/scr"
    # Constructing load_name for training
    unique_id = args.UNIQUE_ID
    task = args.task

    load_name = (
        f"{prefix_path}/echollm/checkpoints/{args.loss_type}/frac{args.dataset_fractions}/{args.model_tag}/"
        f"{task}M{args.stage - 1}-"
        f"{model_parameter[args.model_tag]}-"
        f"batch{args.batch_size}-"
        f"lr{args.learning_rate}-"
        f"gamma{args.focal_gamma}-"
        f"seed{args.seed}_{unique_id}/best.ckpt"
    )
    save_name = (
        f"{prefix_path}/echollm/checkpoints/{args.loss_type}/frac{args.dataset_fractions}/{args.model_tag}/"
        f"{task}M{args.stage}-"
        f"{model_parameter[args.model_tag]}-"
        f"batch{args.batch_size}-"
        f"lr{args.learning_rate}-"
        f"gamma{args.focal_gamma}-"
        f"seed{args.seed}_{unique_id}"
    )
    if stage == 0:
        load_name = ""
    print(f"Saving the model at {save_name}")
    # Prepare train_args
    train_args = {
        "tag_model": args.model_tag,
        "loss_type": args.loss_type,
        "pretrained": True,
        "load_name": load_name,
        "load_generate": args.load_generate,
        "batch_size": args.batch_size,
        "num_devices": args.num_devices,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "num_workers": args.num_workers,
        "block_size": args.block_size,
        "max_epochs": args.max_epochs,
        "strategy": args.strategy,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "blimp_eval": args.blimp_eval.split(','),
        "evaluate_KT": args.evaluate_kt,
        "stage": stage,
        "dataset_fractions": args.dataset_fractions,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "save_name": save_name,
        "version_name": (
            f"Pedram/M{stage}/{args.model_tag}-{args.dataset_fractions}-batch{args.batch_size}-"
            f"lr{args.learning_rate}-{args.loss_type}-seed{args.seed}-{unique_id}"
        ),
        "test_mode": args.test_mode,
        "tags": args.tags.split(','),
        "task": args.task,
        "scale": args.scale,
    }

    # Run training
    print(f"--------------- Running training for stage {stage} ---------------")
    train_func(train_args)
    os.system(f"chmod -R 777 {train_args['save_name']}")
    
    if args.server == "aws":
        # Connect to S3
        s3_client = connect_to_s3(
            access_key=args.AWS_ACCESS_KEY,
            secret_key=args.AWS_SECRET_KEY,
            region_name=args.AWS_REGION,
        )

        print("Uplloading checkpoints to S3")
        s3_checkpoint_path = (f"checkpoints/{args.loss_type}/frac{args.dataset_fractions}/{args.model_tag}/"
                              f"M{args.stage}-"
                              f"{model_parameter[args.model_tag]}-"
                              f"batch{args.batch_size}-"
                              f"lr{args.learning_rate}-"
                              f"seed{args.seed}/")
        # Upload the models to S3
        upload_folder(s3_client, args.bucket_name, train_args["save_name"], s3_prefix=s3_checkpoint_path)

    generation_args = {
        "task": args.task,
        "tag_model": args.model_tag,
        "pretrained": True,
        "load_name": f"{train_args['save_name']}/best.ckpt",
        "batch_size": args.generate_batch_size,
        "num_workers": args.generate_num_workers,
        "block_size": 32,
        "total_fractions": args.dataset_fractions,
        "load_generate": args.load_generate,
        "generate": f"{args.load_generate}/stage{stage}.txt",
        "generated_length": args.generated_length,
        "generation_context_length": args.generation_context_length,
        "use_consistency_block": False,
        "stage": stage,
        "task": args.task,
    }

    # Run generation
    print(f"--------------- Generating for stage {stage} ---------------")
    generate_func(generation_args)
    os.system(f"chmod -R 777 {generation_args['generate']}")

    if args.server == "aws":
        print("Uploading generated datasets to S3")

        s3_generate_path = (f"generated_datasets/{args.model_tag}/{args.loss_type}/frac{args.dataset_fractions}/"
                            f"batch{args.batch_size}-"
                            f"lr{args.learning_rate}-"
                            f"seed{args.seed}/")
        # Upload the models to S3
        upload_folder(s3_client, args.bucket_name, args.load_generate, s3_prefix=s3_generate_path)

def main():
    parser = argparse.ArgumentParser(description="Run training and generation stages.")
    
    # Define all arguments
    parser.add_argument("--stage", type=int, required=True, help="Stage number")
    parser.add_argument("--model_tag", type=str, required=True, help="Model tag")
    parser.add_argument("--loss_type", type=str, help="Loss type", default="baseline")
    parser.add_argument("--focal_gamma", type=float, help="Focal gamma", default=None)
    parser.add_argument("--focal_alpha", type=float, help="Focal alpha", default=None)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="Accumulate gradient batches")
    parser.add_argument("--num_workers", type=int, default=127, help="Number of workers")
    parser.add_argument("--block_size", type=int, default=512, help="Block size")
    parser.add_argument("--max_epochs", type=int, default=10, help="Max epochs")
    parser.add_argument("--strategy", type=str, default="ddp", help="Strategy")
    parser.add_argument("--seed", type=int, default=10, help="Seed")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--blimp_eval", type=str, default="adjunct_island,causative", help="Blimp eval")
    parser.add_argument("--evaluate_kt", type=str, default="kt_dataset_gpt_mini.json", help="Evaluate KT")
    parser.add_argument("--dataset_fractions", type=int, required=True, help="Dataset fractions")
    parser.add_argument("--load_generate", type=str, required=True, help="Load generate path")
    parser.add_argument("--generated_length", type=int, default=32, help="Generated length")
    parser.add_argument("--generation_context_length", type=int, default=512, help="Generation context length")
    parser.add_argument("--use_consistency_block", type=bool, default=False, help="Use consistency block")
    parser.add_argument("--generate_batch_size", type=int, default=16, help="Generate batch size")
    parser.add_argument("--generate_num_workers", type=int, default=64, help="Generate num workers")
    parser.add_argument("--test_mode", type=bool, default=False, help="Train without evaluation")
    parser.add_argument("--tags", type=str, default="", help="Tags for W&B")
    parser.add_argument("--task", type=str, default="wikitext", help="Training task")
    parser.add_argument("--scale", type=int, default=0, help="Scale of synthetic data for short experiments")

    parser.add_argument("--server", type=str, default="local", help="Server to run on")
    parser.add_argument("--AWS_ACCESS_KEY", required=True, help="AWS access key ID.")
    parser.add_argument("--AWS_SECRET_KEY", required=True, help="AWS secret access key.")
    parser.add_argument("--AWS_REGION", default="us-west-2", help="AWS region.")
    parser.add_argument("--bucket-name", default="aceslab", help="S3 bucket name.")
    parser.add_argument("--UNIQUE_ID", required=True, help="Unique time id for each run")

    # Parse arguments
    args = parser.parse_args()

    # Run specified stage
    run_stage(args)

if __name__ == "__main__":
    main()
