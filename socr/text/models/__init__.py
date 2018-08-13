from .resSru import resSru

def get_model_by_name(name):
    return {
        "resSru": resSru,
}[name]