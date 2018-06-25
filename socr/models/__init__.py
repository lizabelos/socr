import torch

from .dilatation_gru_network import DilatationGruNetwork
from .bigger_conv_lstm_network import BiggerConvLSTMNetwork
from .lstm_2d_localization_network import LSTM2DLocalizationNetwork
from .ocropy_line_network import OcropyLineNetwork
from .resnet_localization import ResnetLocalization
from .conv_lstm_network import ConvLSTMNetwork
from .x_height_labeling_model import XHeightLabelingModel
from .x_height_resnet_model import XHeightResnetModel


def get_model_by_name(name):
    return {
        "ResnetLocalization": ResnetLocalization,
        "ConvLSTMNetwork": ConvLSTMNetwork,
        "BiggerConvLSTMNetwork": BiggerConvLSTMNetwork,
        "OcropyLineNetwork": OcropyLineNetwork,
        "LSTM2DLocalizationNetwork": LSTM2DLocalizationNetwork,
        "XHeightLabelingModel": XHeightLabelingModel,
        "DilatationGruNetwork": DilatationGruNetwork,
        "XHeightResnetModel": XHeightResnetModel
    }[name]


def get_optimizer_by_name(name):
    return {
        "SGD": torch.optim.SGD,
        "RMSProp": torch.optim.RMSprop,
        "Adam": torch.optim.Adam
    }[name]
