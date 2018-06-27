import os
import subprocess
import sys
import git


def build_wrapctc():
    print("You need wrapctc library to continue. Do you want to install it ? [yes/no]")
    response = input()

    if response == "yes":
        os.makedirs('submodules/warp-ctc', exist_ok=True)
        git.Git("submodules").clone("https://github.com/t-vi/warp-ctc.git")
        res = subprocess.run([sys.executable, 'setup.py', 'build'], cwd='submodules/warp-ctc/pytorch_binding')
        assert res.returncode == 0, "Error"
        res = subprocess.run([sys.executable, 'setup.py', 'install'], cwd='submodules/warp-ctc/pytorch_binding')
        assert res.returncode == 0, "Error"
    else:
        print("Goodbye :(")
        exit(0)


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
