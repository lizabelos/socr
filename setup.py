from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

extensions = [
    Extension("socr.models.decoders.baseline_decoder", ["socr/models/decoders/baseline_decoder.py"]),
    Extension("socr.utils.rating.word_error_rate", ["socr/utils/rating/word_error_rate.py"])
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
)