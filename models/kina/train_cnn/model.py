import torch as t
import torch.nn as nn
import torchvision as tv
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.feature_extraction import create_feature_extractor

class PatchModel(nn.Module):
    def __init__(self):
        super(PatchModel, self).__init__()
        
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        #self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        # self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.features[-2:].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(576, 1)
        # )
        
        # self.model = self.model.cuda().float()

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 1)
        )

        # assuming 64x64 input
        # self.model = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.MaxPool2d(2, 2),
        #     nn.Flatten(),
        #     nn.Linear(64*32*32, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(1024, 1)
        # )
        
        self.model = self.model.cuda().float()


    def forward(self, x):
        #features = self.model(x)
        #features = features.view(features.size(0), -1)
        #return self.classifier(features)
        return self.model(x)


class MobileNetV3_FPN_Model(nn.Module):
    def __init__(self):
        super(MobileNetV3_FPN_Model, self).__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features
        layers = ['0', '1', '2', '9']

        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers]):
                parameter.requires_grad_(False)

        in_channels_list = [backbone[int(i)].out_channels for i in layers]
        return_nodes = {layer: f'{index}' for index, layer in enumerate(layers)}

        self.out_channels = 256
        self.extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

        # assuming 300x300 input
        layer_shapes = [150, 75, 38, 10, 5]
        
        self.featurizers = nn.ModuleList()
        for i in range(5):
            self.featurizers.append(nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(64 * layer_shapes[i]**2, 64)
            ))

        self.classifier = nn.Linear(len(layer_shapes) * 64, 1)

        self.to_gpu()


    def to_gpu(self):
        self.extractor = self.extractor.cuda().float()
        self.fpn = self.fpn.cuda().float()
        self.featurizers = self.featurizers.cuda().float()
        self.classifier = self.classifier.cuda().float()


    def forward(self, x):
        x = self.extractor(x)
        x = self.fpn(x)
        
        output = []
        for i, feat in enumerate(x):
            output.append(self.featurizers[i](x[feat]))
        
        output = t.cat(output, dim=1)
        output = self.classifier(output)

        return output


class EfficientNetB0_Model(nn.Module):
    def __init__(self):
        super(EfficientNetB0_Model, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.features[-1:].parameters():
            param.requires_grad = True

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 1)
        )
        
    def forward(self, x):
        return self.model(x)


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # assuming 300x300 input
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64*150*150, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.model = self.model.cuda().float()
        self.classifier = self.classifier.cuda().float()
        
        # warm up
        self.eval()
        input = t.randn(1, 3, 300, 300).cuda().float()
        self.forward(input)
        input.cpu()


    def forward(self, x):
        features = self.model(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)