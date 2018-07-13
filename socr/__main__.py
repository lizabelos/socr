import sys

from socr.utils.logging.logger import print_normal
from .line_localizator import main as line_localizator_main
from .text_recognizer import main as text_recognizer_main
from .text_generator import main as generator_main
from .recognizer import main as recognizer_main


def print_help():
    print_normal("socr [line|text|generator|recognize]")


if len(sys.argv) <= 1:
    print_help()
elif sys.argv[1] == "line":
    line_localizator_main(sys.argv[2:])
elif sys.argv[1] == "text":
    text_recognizer_main(sys.argv[2:])
elif sys.argv[1] == "generator":
    generator_main(sys.argv[2:])
elif sys.argv[1] == "recognizer":
    recognizer_main(sys.argv[2:])
else:
    print_help()
