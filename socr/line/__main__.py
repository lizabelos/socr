import argparse

from socr.utils import download_resources
from socr.line import LineLocalizator


def main():
    parser = argparse.ArgumentParser(description="socr")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--model', type=str, default="dhSegment", help="Model name")
    parser.add_argument('--execute', type=str, default=None)
    parser.add_argument('--evaluate', type=str, default=None)
    parser.add_argument('--evaluatehist', type=str, default=None)
    parser.add_argument('--generateandexecute', action='store_const', const=True, default=False)
    parser.add_argument('--test', action='store_const', const=True, default=False)
    parser.add_argument('--testgenerator', action='store_const', const=True, default=False)
    parser.add_argument('--disablecuda', action='store_const', const=True, default=False)
    parser.add_argument('--overlr', action='store_const', const=True, default=False)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    download_resources()
    line_recognizer = LineLocalizator(args.model, args.lr, args.name, not args.disablecuda)

    if args.testgenerator:
        line_recognizer.test_generator()
    elif args.generateandexecute:
        line_recognizer.eval()
        line_recognizer.generateandexecute()
    elif args.execute is not None:
        line_recognizer.eval()
        line_recognizer.execute(args.execute)
    elif args.evaluate is not None:
        line_recognizer.eval()
        line_recognizer.evaluate(args.evaluate)
    elif args.evaluatehist is not None:
        line_recognizer.eval()
        line_recognizer.evaluate_hist(args.evaluatehist)
    elif args.test:
        line_recognizer.eval()
        line_recognizer.callback()
    else:
        if args.overlr:
            line_recognizer.train(args.bs, args.lr)
        else:
            line_recognizer.train(args.bs)

main()