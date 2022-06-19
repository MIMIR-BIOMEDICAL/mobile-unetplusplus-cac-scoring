import os
import subprocess

from dotenv import load_dotenv

load_dotenv()


def install_az():
    print("AzCopy not found, installing AzCopy")
    subprocess.run(["./install_azcopy.sh"], text=True)
    print("Azcopy Installed")


def download_coca_dataset():
    print("Downloading Dataset")
    subprocess.run(["azcopy", "copy", os.getenv("DATASET_URL"), "data", "--recursive"], text=True)
    print("Dataset Downloaded")


def clean_directory():
    print("Cleaning Directory")
    subprocess.run(["mv", "data/cocacoronarycalciumandchestcts-2/Gated_release_final", "dataset/"])
    subprocess.run(["rm", "-r", "data"])
    print("Finish Cleaning Directory")


def main():
    if "dataset" not in os.listdir("."):
        try:
            download_coca_dataset()
            clean_directory()
        except:
            install_az()
            download_coca_dataset()
            clean_directory()
    else:
        print("Dataset in directory")


if __name__ == "__main__":
    main()
