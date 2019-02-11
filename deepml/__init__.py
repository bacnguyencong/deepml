from . import losses
from . import datasets

# list of all arquitechtures
MODEL_NAMES = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'bnincepnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
    'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]
# list of all losses
MODEL_LOSSES = sorted(
    name for name in losses.__dict__
    if callable(losses.__dict__[name]) and not name.startswith("__")
)
# list of all data sets
MODEL_DATASETS = sorted(
    name for name in datasets.__dict__
    if callable(datasets.__dict__[name]) and not name.startswith("__")
)
