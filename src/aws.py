import argparse
import boto3
import os

from botocore.exceptions import NoCredentialsError, PartialCredentialsError

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
        s3_client.download_file(bucket_name, object_key, download_path)
        print(f"File '{object_key}' downloaded successfully to '{download_path}'.")
    except Exception as e:
        print(f"Failed to download file '{object_key}':", e)

def upload_file(s3_client, bucket_name: str, file_path: str, object_key: str):
    """
    Upload a file to an S3 bucket.

    Parameters:
        s3_client (boto3.client): A boto3 S3 client instance.
        bucket_name (str): Name of the S3 bucket.
        file_path (str): Local file path to upload.
        object_key (str): Key for the file in the S3 bucket.
    """
    try:
        s3_client.upload_file(file_path, bucket_name, object_key)
        print(f"File '{file_path}' uploaded successfully as '{object_key}' to bucket '{bucket_name}'.")
    except Exception as e:
        print(f"Failed to upload file '{file_path}':", e)

def upload_folder(s3_client, bucket_name, folder_path, s3_prefix=""):
    """
    Uploads an entire folder along with its subfolders to an S3 bucket.
    
    :param bucket_name: Name of the S3 bucket
    :param folder_path: Local path of the folder to upload
    :param s3_prefix: Prefix in the S3 bucket (optional)
    """

    # Normalize folder path
    folder_path = os.path.abspath(folder_path)
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, folder_path)
            s3_key = os.path.join(s3_prefix, relative_path)

            print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(local_file_path, bucket_name, s3_key)


# Example usage:
if __name__ == "__main__":
    # Input args
    parser = argparse.ArgumentParser(description="Download and upload files to S3.")
    parser.add_argument("--access-key", required=True, help="AWS access key ID.")
    parser.add_argument("--secret-key", required=True, help="AWS secret access key.")
    parser.add_argument("--region", default="us-west-2", help="AWS region.")
    parser.add_argument("--bucket-name", required=True, help="S3 bucket name.")
    parser.add_argument("--file-path", default="file.txt", help="Local path to save downloaded file.")
    parser.add_argument("--upload-path", default="file.txt", help="Local path to upload file.")


    args = parser.parse_args()


    ACCESS_KEY = args.access_key
    SECRET_KEY = args.secret_key
    REGION = args.region

    BUCKET_NAME = args.bucket_name
    
    # File paths
    LOCAL_PATH = args.file_path
    UPLOAD_PATH = args.upload_path
    # OBJECT_KEY_DOWNLOAD = "file-in-s3.txt"
    # OBJECT_KEY_UPLOAD = "uploaded-file-in-s3.txt"

    # Connect to S3
    s3 = connect_to_s3(ACCESS_KEY, SECRET_KEY, REGION)

    if s3:
        # Download file
        # download_file(s3, BUCKET_NAME, OBJECT_KEY_DOWNLOAD, LOCAL_DOWNLOAD_PATH)
        # Upload file
        # upload_file(s3, BUCKET_NAME, LOCAL_PATH, UPLOAD_PATH)
        # Upload folder
        upload_folder(s3, BUCKET_NAME, LOCAL_PATH, s3_prefix=UPLOAD_PATH)
