from .dhSegment import dhSegment


def get_model_by_name(name):
    return {
        "dhSegment": dhSegment,
}[name]