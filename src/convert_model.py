import argparse
import boto3
import os

import pytorch_lightning as pl

from transformers import AutoTokenizer, AutoModelForCausalLM
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from plt_model import LitModel

def connect_to_s3(access_key: str, secret_key: str, region_name: str = 'us-east-1'):
    """
    Establish a connection to S3 using provided credentials.

    Parameters:
        access_key (str): AWS access key ID.
        secret_key (str): AWS secret access key.
        region_name (str): AWS region (default: 'us-east-1').

    Returns:
        s3_client (boto3.client): A boto3 S3 client instance.
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        print("Connected to S3 successfully.")
        return s3_client
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("Error in connecting to S3:", e)
        return None

def download_file(s3_client, bucket_name: str, object_key: str, download_path: str):
    """
    Download a file from an S3 bucket.

    Parameters:
        s3_client (boto3.client): A boto3 S3 client instance.
        bucket_name (str): Name of the S3 bucket.
        object_key (str): Key of the object to download.
        download_path (str): Local path to save the downloaded file.
    """
    try:
        if not os.path.exists(os.path.dirname(download_path)):
            os.makedirs(os.path.dirname(download_path))
        # Download the file
        print(f"Downloading '{object_key}' from bucket '{bucket_name}' to '{download_path}'...")
        s3_client.download_file(bucket_name, object_key, download_path)
        print(f"File '{object_key}' downloaded successfully to '{download_path}'.")
    except Exception as e:
        print(f"Failed to download file '{object_key}':", e)

# Example usage:
if __name__ == "__main__":
    # Input args
    parser = argparse.ArgumentParser(description="Download and upload files to S3.")
    parser.add_argument("--access-key", default="", help="AWS access key ID.")
    parser.add_argument("--secret-key", default="", help="AWS secret access key.")
    parser.add_argument("--region", default="us-west-2", help="AWS region.")
    parser.add_argument("--bucket-name", default="aceslab", help="S3 bucket name.")
    parser.add_argument("--local-path", default="/scr/echollm/checkpoints", help="Local path to save downloaded file.")
    parser.add_argument("--aws-path", default="/scr/echollm/checkpoints", help="Local path to upload file.")
    parser.add_argument("--token", default="hf_token", help="Hugging Face token.")
    parser.add_argument("--server", default="local", help="Server type (local or aws).")
    args = parser.parse_args()

    LOCAL_PATH = args.local_path
  
    if args.server == "local":
        
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt",
                                                  cache_dir='./model_cache_dir',
                                                  return_dict=True,
                                                  token=args.token,
                                                 )        
        tokenizer.pad_token = tokenizer.eos_token
        # Checking if the model already exists
        print(os.path.exists(LOCAL_PATH + "/config.json"), LOCAL_PATH + "/config.json")
        if not os.path.exists(LOCAL_PATH + "/config.json"):   
            plt_model = LitModel.load_from_checkpoint(LOCAL_PATH + "/last.ckpt", map_location="cpu", tokenizer=tokenizer)
            plt_model.model.save_pretrained(LOCAL_PATH)
            plt_model.tokenizer.save_pretrained(LOCAL_PATH)
            print("Model saved in HF format successfully.")
        else:
            print("Model already exists. Skipping saving the HF checkpoints.")
    else:
        ACCESS_KEY = args.access_key
        SECRET_KEY = args.secret_key
        REGION = args.region
        BUCKET_NAME = args.bucket_name
        # File paths
        SAVE_PATH = LOCAL_PATH[:-9]
        AWS_PATH = args.aws_path

        # Connect to S3
        
        s3 = connect_to_s3(ACCESS_KEY, SECRET_KEY, REGION)
        if s3:
            if not os.path.exists(os.path.dirname(LOCAL_PATH)):
                download_file(s3, BUCKET_NAME, AWS_PATH, LOCAL_PATH)
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt",
                                                        cache_dir='./model_cache_dir',
                                                        return_dict=True,
                                                        token=args.token,
                                                        )        
                tokenizer.pad_token = tokenizer.eos_token

                plt_model = LitModel.load_from_checkpoint(LOCAL_PATH, map_location="cpu", tokenizer=tokenizer)
                plt_model.model.save_pretrained(SAVE_PATH)
                plt_model.tokenizer.save_pretrained(SAVE_PATH)
            else:
                print("Model already exists. Skipping download and saving the HF checkpoints.")
    
       
