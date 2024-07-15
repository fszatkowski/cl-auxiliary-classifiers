from torchvision import models

from .lenet import LeNet
from .resnet32 import resnet32
from .convnext_cifar import *
from .resnet_cifar import *
from .vggnet import VggNet

# available torchvision models
tvmodels = [
    "alexnet",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "googlenet",
    "inception_v3",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]

allmodels = tvmodels + [
    resnet_cifar.__all__,
    convnext_cifar.__all__,
    "resnet32",
    "LeNet",
    "VggNet",
]


def set_tvmodel_head_var(model):
    if type(model) == models.AlexNet:
        model.head_var = "classifier"
    elif type(model) == models.DenseNet:
        model.head_var = "classifier"
    elif type(model) == models.EfficientNet:
        model.head_var = "classifier"
    elif type(model) == models.Inception3:
        model.head_var = "fc"
    elif type(model) == models.ResNet:
        model.head_var = "fc"
    elif type(model) == models.VGG:
        model.head_var = "classifier"
    elif type(model) == models.GoogLeNet:
        model.head_var = "fc"
    elif type(model) == models.MobileNetV2:
        model.head_var = "classifier"
    elif type(model) == models.ShuffleNetV2:
        model.head_var = "fc"
    elif type(model) == models.SqueezeNet:
        model.head_var = "classifier"
    elif type(model) == models.MobileNetV3:
        model.head_var = "classifier"
    elif type(model) == models.VisionTransformer:
        model.head_var = "heads"
    elif type(model) == models.ConvNeXt:
        model.head_var = "classifier"
    else:
        raise ModuleNotFoundError
