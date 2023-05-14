"""CLI for gcs download and upload"""
import os
import sys

import inquirer
from google.cloud import storage
from google.oauth2 import service_account
from tqdm import tqdm


# Function to get credentials
def get_credentials(creds_path):
    """
    Reads Google Cloud credentials from a service account file and returns them as
    a Credentials object that can be used to authenticate with Google Cloud services.

    Args:
    - creds_path (str): The path to the service account file containing the credentials.

    Returns:
    - The credentials (Credentials): A Credentials object that can be used to authenticate
      with Google Cloud services.

    Raises:
    - An exception if there was an error reading the credentials or if the credentials
      were not found at the given path.
    """
    creds = None
    try:
        creds = service_account.Credentials.from_service_account_file(creds_path)
    except Exception as e:
        print(f"Error getting credentials: {e}")

    return creds


def upload_to_gcs(credentials_obj, bucket_name, local_file_path, destination_blob_name):
    """
    Uploads a file to Google Cloud Storage (GCS).

    Args:
        credentials: The credentials used to authenticate with GCS.
        bucket_name: The name of the bucket to upload the file to.
        local_file_path: The local path to the file to be uploaded.
        destination_blob_name: The name of the file in GCS after it has been uploaded.

    Returns:
        None.

    Raises:
        Exception: If there was an error uploading the file to GCS.

    """
    try:
        storage_client = storage.Client(credentials=credentials_obj)
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            bucket = storage_client.create_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        with open(local_file_path, "rb") as in_file:
            total_bytes = os.fstat(in_file.fileno()).st_size
            with tqdm.wrapattr(
                in_file,
                "read",
                total=total_bytes,
                miniters=1,
                desc=f"upload to {bucket_name}",
            ) as file_obj:
                blob.upload_from_file(
                    file_obj,
                    content_type=None,
                    size=total_bytes,
                )

        print(
            f"File {local_file_path} uploaded to {bucket_name} as {destination_blob_name}"
        )
    except Exception as e:
        print(f"Error uploading file to GCS: {e}")


def download_from_gcs(credentials_obj, bucket_name, source_blob_name, local_file_path):
    """
    Downloads a file from Google Cloud Storage to a local file.

    Args:
    - credentials (google.auth.credentials.Credentials): Google Cloud Service Account credentials.
    - bucket_name (str): The name of the bucket where the file is stored.
    - source_blob_name (str): The name of the file to be downloaded.
    - local_file_path (str): The local path where the file will be downloaded to.

    Returns:
    None

    Raises:
    - Exception: If there is an error during the download process.

    """
    try:
        storage_client = storage.Client(credentials=credentials_obj)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(local_file_path)

        print(
            f"File {source_blob_name} downloaded from {bucket_name} to {local_file_path}"
        )
    except Exception as e:
        print(f"Error downloading file from GCS: {e}")


if __name__ == "__main__":
    arg = sys.argv
    if len(arg) == 2 and arg[1] == "-n":
        answer = {"action": "Download", "creds_path": "serviceAccount.json"}
    else:
        # Ask user for action
        questions = [
            inquirer.Text(
                "creds_path",
                message="GCP Credentials path?",
                default="serviceAccount.json",
            ),
            inquirer.List(
                "action",
                message="What do you want to do?",
                choices=["Upload", "Download"],
            ),
        ]
        answer = inquirer.prompt(questions)

    # Get credentials
    credentials = get_credentials(answer["creds_path"])

    # Upload file to GCS
    if answer["action"] == "Upload":
        # Ask for bucket name and file path
        questions = [
            inquirer.Text("bucket_name", message="Enter bucket name:"),
            inquirer.Text("local_file_path", message="Enter local file path:"),
            inquirer.Text(
                "destination_blob_name", message="Enter destination blob name:"
            ),
        ]
        answers = inquirer.prompt(questions)

        cli_bucket_name = answers["bucket_name"]
        cli_local_file_path = answers["local_file_path"]
        cli_destination_blob_name = answers["destination_blob_name"]

        # Upload file to GCS
        upload_to_gcs(
            credentials, cli_bucket_name, cli_local_file_path, cli_destination_blob_name
        )

    # Download file from GCS
    elif answer["action"] == "Download":
        if len(arg) == 2 and arg[1] == "-n":
            answers = {
                "bucket_name": "mobile-unet-bucket",
                "source_blob_name": "dataset",
                "local_file_path": "data/processed.tar.gz",
            }

        else:
            # Ask for bucket name and file path
            questions = [
                inquirer.Text("bucket_name", message="Enter bucket name:"),
                inquirer.Text("source_blob_name", message="Enter source blob name:"),
                inquirer.Text("local_file_path", message="Enter local file path:"),
            ]
            answers = inquirer.prompt(questions)

        cli_bucket_name = answers["bucket_name"]
        cli_source_blob_name = answers["source_blob_name"]
        cli_local_file_path = answers["local_file_path"]

        # Download file from GCS
        download_from_gcs(
            credentials, cli_bucket_name, cli_source_blob_name, cli_local_file_path
        )
