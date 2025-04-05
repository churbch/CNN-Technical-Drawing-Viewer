import torch
import torch.nn as nn
import torchvision
from torchvision.ops import FeaturePyramidNetwork

class PartsDetector(nn.Module):
    def __init__(self, num_classes):
        super(PartsDetector, self).__init__()
        # ResNet50 backbone for feature extraction
        backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        backbone_out_channels = [256, 512, 1024, 2048]
        # FPN used to generate feature maps at different scales
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=backbone_out_channels,
            out_channels=256
        )

        self.detect_head = nn.ModuleList([
            self._make_detection_head(256, num_classes) for _ in range(4)
        ])

        self.completeness_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
    # Detection head for each feature map, predicting bounding boxes and class scores
    def _make_detection_head(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4 + num_classes, kernel_size=1)
        )
    # Forward uses the backbone to extract features, FPN to generate multi-scale feature maps,
    # and detection heads to predict bounding boxes and class scores
    def forward(self, x):
        c2 = self.backbone[4](self.backbone[:5](x))
        c3 = self.backbone[5](c2)
        c4 = self.backbone[6](c3)
        c5 = self.backbone[7](c4)

        fpn_feats = self.fpn({'0': c2, '1': c3, '2': c4, '3': c5})
        predictions = [head(feat) for head, feat in zip(self.detect_head, fpn_feats.values())]
        completeness = self.completeness_head(c5)

        return predictions, completeness
