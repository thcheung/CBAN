from .multiNet import MultiNet
from .textNet import TextNet
from .imageNet import ImageNet
from .cbanNet import CbanNet

def get_model(model_name, hidden_dim, classes, dropout, language):
    if model_name == 'multi':
        return MultiNet(hidden_dim, classes,
                            dropout, language=language)

    if model_name == 'text':
        return TextNet(hidden_dim, classes,
                            dropout, language=language)

    if model_name == 'cban':
        return CbanNet(hidden_dim, classes,
                            dropout, language=language)

    if model_name == 'image':
        return ImageNet(hidden_dim, classes,
                            dropout, language=language)