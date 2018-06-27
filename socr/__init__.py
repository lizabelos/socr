import torch

from socr.dataset.set.oriented_document_generated_set import OrientedDocumentGeneratedSet
from socr.utils.logging.logger import print_normal, print_error

print_normal("Using torch version " + torch.__version__)

torch_version = [int(i) for i in torch.__version__.split(".")]
if torch_version[0] == 0 and torch_version[1] < 4:
    print_error("Your torch version is incompatible with SOCR. Required is 0.4.0")
    exit(0)

from .line_localizator import LineLocalizator
from .text_recognizer import TextRecognizer
from socr.recognizer import Recognizer