from configparser import ConfigParser

from socr.dataset.generator.document_generator_helper import DocumentGeneratorHelper
from socr.dataset.generator.line_generator import LineGenerator
from socr.dataset.set.line_generated_set import LineGeneratedSet
from socr.dataset.set.oriented_document_generated_set import OrientedDocumentGeneratedSet
from socr.dataset.set.iam_handwriting_line_database import IAMHandwritingLineDatabase
from socr.dataset.set.iam_oneline_handwriting_database import IAMOneLineHandwritingDatabase
from socr.dataset.set.iam_washington import IAMWashington
from socr.dataset.set.icdar_document_set import ICDARDocumentSet
from socr.dataset.set.merged_set import MergedSet


def parse_datasets_configuration_file(helper, path="datasets.cfg", with_document=False, with_line=False, training=False, testing=False, args=None):

    config = ConfigParser()
    config.read(path)

    datasets = []

    for section in config.sections():
        dict = {}
        options = config.options(section)
        for option in options:
            dict[option] = config.get(section, option)

        if dict["for"] != "Document" and dict["for"] != "Line":
            print("Invalid for : '" + dict["for"] + "'")

        if dict["for"] == "Document" and with_document == False:
            continue

        if dict["for"] == "Line" and with_line == False:
            continue

        if training and "train" in dict:
            print("Loading " + str(dict) + "...")
            dataset = parse_dataset(helper, dict["type"], dict["train"], args)
            if dataset is not None:
                datasets.append(dataset)

        if testing and "test" in dict:
            print("Loading " + str(dict) + "...")
            dataset = parse_dataset(helper, dict["type"], dict["test"], args)
            if dataset is not None:
                datasets.append(dataset)

    if len(datasets) == 0:
        return None

    merged_datasets = datasets[0]
    for i in range(1, len(datasets)):
        merged_datasets = MergedSet(merged_datasets, datasets[i])

    return merged_datasets


def parse_dataset(helper, type, path, args=None):

    if args is None:
        args = {}

    if type == "DocumentGenerator":
        if path == "yes":
            return OrientedDocumentGeneratedSet(helper, **args)
        else:
            return None

    if type == "LineGenerator":
        if path == "yes":
            return LineGeneratedSet(helper, **args)
        else:
            return None

    if type == "ICDAR-Baseline":
        return ICDARDocumentSet(helper, path, **args)

    if type == "IAM":
        return IAMHandwritingLineDatabase(helper, path, **args)

    if type == "IAM-One-Line":
        return IAMOneLineHandwritingDatabase(helper, path, **args)

    if type == "IAM-Washington":
        return IAMWashington(helper, path, **args)

    print("Warning : unknown database type : '" + type + "'")

    return None
