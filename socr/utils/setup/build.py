import os
import subprocess
import sys
from shutil import copyfile

import git

from socr.utils.logging.logger import print_warning


def build_wrapctc():
    print("You need wrapctc library to continue. Do you want to install it ? [yes/no]")
    response = input()

    if response == "yes":
        my_env = os.environ.copy()
        my_env["CXX"] = "g++-5"
        my_env["CMAKE_CXX_COMPILER"] = "g++5"
        my_env["CC"] = "gcc-5"
        my_env["CMAKE_C_COMPILER"] = "gcc-5"

        os.makedirs('submodules/warp-ctc', exist_ok=True)
        git.Git("submodules").clone("https://github.com/t-vi/warp-ctc.git")
        res = subprocess.run([sys.executable, 'setup.py', 'build'], cwd='submodules/warp-ctc/pytorch_binding',
                             env=my_env)
        assert res.returncode == 0, "Error"
        res = subprocess.run([sys.executable, 'setup.py', 'install'], cwd='submodules/warp-ctc/pytorch_binding',
                             env=my_env)
        assert res.returncode == 0, "Error"
    else:
        print("Goodbye :(")
        exit(0)


def install_and_import_wrapctc():
    import importlib
    try:
        importlib.import_module('warpctc')
    except ImportError:
        build_wrapctc()
    finally:
        globals()['wrapctc'] = importlib.import_module('wrapctc')


def build_sru():
    print("You need sru library to continue. Do you want to install it ? [yes/no]")
    response = input()

    if response == "yes":
        os.makedirs('submodules/sru', exist_ok=True)
        git.Git("submodules").clone("https://github.com/taolei87/sru.git")
        res = subprocess.run([sys.executable, 'setup.py', 'install'], cwd='submodules/sru')
        assert res.returncode == 0, "Error"
    else:
        print("Goodbye :(")
        exit(0)


def install_and_import_sru():
    import importlib
    try:
        importlib.import_module('sru')
    except ImportError:
        build_wrapctc()
    finally:
        globals()['sru'] = importlib.import_module('sru')


def build_ctcdecode():
    print("You need ctcdecode library to continue. Do you want to install it ? [yes/no]")
    response = input()

    if response == "yes":
        os.makedirs('submodules/ctcdecode', exist_ok=True)
        git.Git("submodules").clone("https://github.com/parlance/ctcdecode.git", recursive=True)
        res = subprocess.run([sys.executable, '-m', 'pip', 'install', '.'], cwd='submodules/ctcdecode')
        assert res.returncode == 0, "Error"
    else:
        print("Goodbye :(")
        exit(0)


def install_and_import_ctcdecode():
    import importlib
    try:
        importlib.import_module('ctcdecode')
    except ImportError:
        build_wrapctc()
    finally:
        globals()['ctcdecode'] = importlib.import_module('ctcdecode')


def load_default_datasets_cfg_if_not_exist():
    if not os.path.isfile("datasets.cfg"):
        print_warning("datasets.cfg not found. Copying datasets.exemple.cfg to datasets.cfg.")
        copyfile("datasets.exemple.cfg", "datasets.cfg")
