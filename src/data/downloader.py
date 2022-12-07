import os
import subprocess
import click
from shutil import which

from dotenv import load_dotenv


@click.command()
@click.option("-u", "--url", help="Dataset URL")
@click.option(
    "-d",
    "--dest",
    default="data/raw/",
    type=click.Path(exists=True),
    help="Destination folder",
)
def download_dataset(url, dest):

    load_dotenv()

    if which("azcopy") is None:
        click.echo("AZCopy is not installed, installing azcopy...")
        subprocess.run(["src/data/install_azcopy.sh"], text=True)
        click.echo("AZCopy installed")

    if url is not None:
        subprocess.run(
            ["azcopy", "copy", url, dest, "--recursive"],
            text=True,
        )
        return

    if os.getenv("DATASET_URL") is not None:
        subprocess.run(
            ["azcopy", "copy", os.getenv("DATASET_URL"), dest, "--recursive"],
            text=True,
        )
        return

    click.echo(
        "Please provide URL via --url options or DATASET_URL environment variable!",
        err=True,
    )


if __name__ == "__main__":
    download_dataset()
