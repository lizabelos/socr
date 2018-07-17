from .abstract_document import AbstractDocument
from .document import Document
from .document_horizontal_layout import DocumentHorizontalLayout
from .document_image import DocumentImage
from .document_text_line import DocumentTextLine
from .document_handwriting_line import DocumentHandwrittingLine
from .document_paragraph import DocumentParagraph
from .document_vertical_layout import DocumentVerticalLayout


def get_document_class_by_name(name):
    return {
        "Document": Document,
        "Paragraph": DocumentParagraph,
        "Textline": DocumentTextLine,
        "Verticallayout": DocumentVerticalLayout,
        "Horizontallayout": DocumentHorizontalLayout,
        "Image": DocumentImage
    }[name]


from .parser import parse_document
