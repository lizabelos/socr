import gzip
from os import listdir
from os.path import join, isfile
from random import randint

from PIL import Image
from lxml import etree

from scribbler.dataset import IAMHandwritingLineDataset

resource_paths = {}
resource_preloaded_image = {}
resource_preloaded_texts = {}


def list_resources(name):
    global resource_paths

    if not name in resource_paths:
        dir_name = "resources/" + name
        resource_paths[name] = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]

    return resource_paths[name]


def preload_image_resources(name):
    global resource_preloaded_image

    if not name in resource_preloaded_image:
        resource = list_resources(name)
        resource_preloaded_image[name] = [Image.open(path).convert('RGB') for path in resource]

    return resource_preloaded_image[name]


def preload_text_ressources(name):
    global resource_preloaded_texts

    if not name in resource_preloaded_texts:
        resources = list_resources(name)
        resource_preloaded_texts[name] = []
        for resource in resources:
            tree = etree.parse(resource)
            root = tree.getroot()
            recursive_preload_text_ressources(name, root)

    return resource_preloaded_texts[name]


def recursive_preload_text_ressources(name, root):
    global resource_preloaded_texts

    title = root.tag.title()
    if title == "Text":
        resource_preloaded_texts[name].append(root.text)
    else:
        for children in root.getchildren():
            recursive_preload_text_ressources(name, children)


def count_resource(name):
    resource = list_resources(name)
    return len(resource)


def peak_resource(name, index):
    resource = list_resources(name)
    return resource[index]


def peak_random_resource(name):
    resource = list_resources(name)
    return resource[randint(0, len(resource) - 1)]


def peak_random_preloaded_image(name):
    resource = preload_image_resources(name)
    return resource[randint(0, len(resource) - 1)]


def peak_random_preloaded_text(name):
    resource = preload_text_ressources(name)
    return resource[randint(0, len(resource) - 1)]


iam_handwriting_line_dataset_instance = None


def get_iam_handwriting_line_dataset_instance():
    global iam_handwriting_line_dataset_instance
    if iam_handwriting_line_dataset_instance is None:
        # TODO : put in a global variable
        iam_handwriting_line_dataset_instance = IAMHandwritingLineDataset("/space_sde/tbelos/dataset/iam-line/train")
    return iam_handwriting_line_dataset_instance
