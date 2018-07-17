from lxml import etree

from scribbler.document import get_document_class_by_name


def parse_document(path):
    tree = etree.parse(path)
    root = tree.getroot()
    result, _ = parse_tree(root)
    return result


def parse_tree(root, parent=None):
    title = root.tag.title()

    root_class = get_document_class_by_name(title)
    root_dict = {"parent": parent}
    x = 0
    y = 0

    for name, value in root.attrib.items():
        if name == "x":
            x = int(value)
        elif name == "y":
            y = int(value)
        else:
            try:
                root_dict[name] = int(value)
            except ValueError:
                root_dict[name] = value

    print(title + " -> " + str(root_dict))

    if title == "Include":
        return parse_tree(etree.parse(root_dict["path"]).getroot())

    root_element = root_class(**root_dict)

    for children in root.getchildren():
        document, position = parse_tree(children, root_element)
        root_element.append_document(document, position)

    return root_element, (x, y)
