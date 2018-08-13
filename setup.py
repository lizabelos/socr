import distutils
import os
import shutil
import subprocess
import sys
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

extensions = [
    Extension("socr.models.decoders.baseline_decoder", ["socr/models/decoders/baseline_decoder.pyx"]),
    Extension("socr.models.decoders.ctc_decoder", ["socr/models/decoders/ctc_decoder.pyx"]),
    Extension("socr.models.encoders.baseline_encoder", ["socr/models/encoders/baseline_encoder.pyx"]),
    Extension("socr.utils.maths.lin_regression", ["socr/utils/maths/lin_regression.pyx"]),
    Extension("socr.utils.rating.word_error_rate", ["socr/utils/rating/word_error_rate.pyx"]),
    Extension("socr.utils.language.beam", ["socr/utils/language/beam.pyx"]),
    Extension("socr.utils.language.prefix_tree", ["socr/utils/language/prefix_tree.pyx"]),
    Extension("socr.utils.language.language_model", ["socr/utils/language/language_model.pyx"]),
    Extension("socr.utils.language.word_beam_search", ["socr/utils/language/word_beam_search.pyx"]),
    Extension("socr.models.loss.ctc", ["socr/models/loss/ctc.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
]


class InstallRequirements(distutils.cmd.Command):

    description = "Install requirements"
    user_options = []

    def initialize_options(self):
        self.cwd = None

    def finalize_options(self):
        self.cwd = os.getcwd()

    def run(self):

        res = subprocess.run(
            [sys.executable, '-m', 'conda', 'install', '-y', 'pytorch=0.4', '-c', 'pytorch'])
        assert res.returncode == 0, "Error"

        res = subprocess.run(
            [sys.executable, '-m', 'conda', 'install', '-y', 'opencv'])
        assert res.returncode == 0, "Error"

        res = subprocess.run(
            [sys.executable, '-m', 'conda', 'install', '-y', '-c', 'conda-forge', 'scikit-image'])
        assert res.returncode == 0, "Error"

        res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        assert res.returncode == 0, "Error"


setup(
    cmdclass={
        'build_ext': build_ext,
        'install_requirements': InstallRequirements,
    },
    ext_modules=cythonize(extensions),
)
