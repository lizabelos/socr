import argparse
import sys

from socr.utils import download_resources
from socr.text import TextRecognizer

def main():
    parser = argparse.ArgumentParser(description="SOCR Text Recognizer")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--model', type=str, default="resSru", help="Model name")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--overlr', action='store_const', const=True, default=False)
    parser.add_argument('--disablecuda', action='store_const', const=True, default=False)
    parser.add_argument('--generateandexecute', action='store_const', const=True, default=False)
    parser.add_argument('--onlyhand', action='store_const', const=True, default=False)
    parser.add_argument('--test', action='store_const', const=True, default=False,
                        help='Test the char generator')
    args = parser.parse_args()
    if args.generateandexecute:
        download_resources()
        line_ctc = TextRecognizer(args.model, args.lr, args.name, not args.disablecuda)
        line_ctc.eval()
        line_ctc.generateandexecute(args.onlyhand)
    elif args.test:
        download_resources()
        line_ctc = TextRecognizer(args.model, args.lr, args.name, not args.disablecuda)
        line_ctc.eval()
        line_ctc.test()
    else:
        download_resources()
        line_ctc = TextRecognizer(args.model, args.lr, args.name, not args.disablecuda)
        if args.overlr:
            line_ctc.train(args.bs, args.lr)
        else:
            line_ctc.train(args.bs)

main()
