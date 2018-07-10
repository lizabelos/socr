import os
import zipfile

import wget

from socr.utils.logging.logger import print_normal


def download_resources():
    if not os.path.isdir("resources/fonts"):
        url = "https://www.dropbox.com/s/3wcp26el8x5na4j/resources.zip?dl=1"

        print_normal("Dowloading resources...")
        wget.download(url)

        print_normal("Extracting resources...")
        zip_ref = zipfile.ZipFile("resources.zip", 'r')
        zip_ref.extractall(".")
        zip_ref.close()

        print_normal("Cleaing up...")
        os.remove("resources.zip")

        print_normal("Resources downloaded successfully.")
