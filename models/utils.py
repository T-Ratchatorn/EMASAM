from .resnet import ResNet18, ResNet50
from .wideresnet import WideResNet
from .resnet_224 import ResNet18_224, ResNet50_224
from .pyramidnet import PyramidNet
from torchvision.models import vit_b_16

def get_model(model_name, data_name, **cfg ):
    name = model_name.lower()
    data = data_name.lower()
    
    if data in ["cifar10", "cifar100"]:
        if( name == "resnet18" ):
            model = ResNet18(cfg['num_classes'], 3)
        elif( name == "resnet50" ):
            model = ResNet50(cfg['num_classes'], 3)
        elif( name == "wideresnet" ):
            model = WideResNet(28, 10, 0, in_channels=3, labels=cfg['num_classes'])
        elif( name == "pyramidnet" ):
            model = PyramidNet(depth=110, alpha=84, in_channels=3, num_classes=cfg['num_classes'])
        else:
            raise NotImplementedError(name)

    elif data in ["fashionmnist","emnist"]:
        if( name == "resnet18" ):
            model = ResNet18(cfg['num_classes'], 1)
        elif( name == "resnet50" ):
            model = ResNet50(cfg['num_classes'], 1)
        elif( name == "wideresnet" ):
            model = WideResNet(28, 10, 0, in_channels=1, labels=cfg['num_classes'])
        elif( name == "pyramidnet" ):
            model = PyramidNet(depth=110, alpha=84, in_channels=1, num_classes=cfg['num_classes'])
        else:
            raise NotImplementedError(name)
            
    elif data in ["flowers102", "oxfordiiitpet", "imagenet_1k"]:
        if( name == "resnet18" ):
            model = ResNet18_224(cfg['num_classes'], 3)
        elif( name == "resnet50" ):
            model = ResNet50_224(cfg['num_classes'], 3)
        elif( name == "vit_b_16" ):
            model = vit_b_16(weights=None, num_classes=cfg['num_classes'])
        else:
            raise NotImplementedError(name)
            
    else:
        raise NotImplementedError(data)

    return model
