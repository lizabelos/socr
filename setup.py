from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

extensions = [
    Extension("socr.models.decoders.baseline_decoder", ["socr/models/decoders/baseline_decoder.pyx"]),
    Extension("socr.models.encoders.baseline_encoder", ["socr/models/encoders/baseline_encoder.pyx"]),
    Extension("socr.utils.maths.lin_regression", ["socr/utils/maths/lin_regression.pyx"]),
    Extension("socr.utils.rating.word_error_rate", ["socr/utils/rating/word_error_rate.pyx"]),
    Extension("socr.utils.language.beam", ["socr/utils/language/beam.pyx"]),
    Extension("socr.utils.language.prefix_tree", ["socr/utils/language/prefix_tree.pyx"]),
    Extension("socr.utils.language.language_model", ["socr/utils/language/language_model.pyx"]),
    Extension("socr.utils.language.word_beam_search", ["socr/utils/language/word_beam_search.pyx"])
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
)