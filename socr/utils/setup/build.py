import os
from shutil import copyfile

from socr.utils.logging.logger import print_warning

def load_default_datasets_cfg_if_not_exist():
    if not os.path.isfile("datasets.cfg"):
        print_warning("datasets.cfg not found. Copying datasets.exemple.cfg to datasets.cfg.")
        copyfile("datasets.default.cfg", "datasets.cfg")
