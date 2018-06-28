from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

extensions = [
    Extension("socr.models.decoders.baseline_decoder", ["socr/models/decoders/baseline_decoder.pyx"]),
    Extension("socr.models.encoders.baseline_encoder", ["socr/models/encoders/baseline_encoder.pyx"]),
    Extension("socr.utils.maths.lin_regression", ["socr/utils/maths/lin_regression.pyx"]),
    Extension("socr.utils.rating.word_error_rate", ["socr/utils/rating/word_error_rate.pyx"])
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
)