from .resRnn import resRnn

def get_model_by_name(name):
    return {
        "resRnn": resRnn,
}[name]