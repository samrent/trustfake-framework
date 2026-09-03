import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from trustfake.models.torch.depth_head import DepthHead

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def _load_pretrained_backbone(model, url):
    """Load an ImageNet checkpoint, skipping the fc layer if num_classes differs.

    The load is non-strict and the report is checked by hand, because
    `strict=False` on its own is exactly the wrong tool: it also swallows a
    checkpoint whose keys do not match the model at all, and the model then
    trains from random weights while claiming to be pretrained. The only keys
    allowed to be missing are the classifier (when its shape differs) and
    an auxiliary depth head, which no ImageNet checkpoint carries. Anything
    else missing, and anything unexpected, is an error.
    """
    state_dict = model_zoo.load_url(url)
    skip_fc = state_dict["fc.weight"].shape[0] != model.fc.weight.shape[0]
    if skip_fc:
        del state_dict["fc.weight"], state_dict["fc.bias"]
    result = model.load_state_dict(state_dict, strict=False)
    allowed = ("depth_head.",) + (("fc.",) if skip_fc else ())
    missing = [k for k in result.missing_keys if not k.startswith(allowed)]
    if missing or result.unexpected_keys:
        msg = (
            f"Pretrained checkpoint {url} does not match the model: "
            f"missing {missing[:5]}{'...' if len(missing) > 5 else ''}, "
            f"unexpected {result.unexpected_keys[:5]}"
            f"{'...' if len(result.unexpected_keys) > 5 else ''}"
        )
        raise RuntimeError(msg)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """torchvision's ResNet, plus an OPT-IN auxiliary depth head (Track C).

    Args:
        block: BasicBlock or Bottleneck.
        layers: Blocks per stage.
        num_classes: Classifier width.
        zero_init_residual: Zero-init the last BN of every residual branch.
        dropout_p: Dropout before the classifier.
        depth_head: Attach a `DepthHead` decoder over the layer3/layer4 maps.
            Off by default, and when off the module tree, the state_dict and
            `forward` are byte-identical to the historical model -- every
            existing checkpoint keeps loading. The head is reached ONLY via
            `forward_with_depth`; `forward` never touches it, so it costs
            nothing at inference and is invisible to every attack.
        depth_head_width: Internal width of the head.
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        dropout_p=0.0,
        depth_head=False,
        depth_head_width=128,
    ):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # Built AFTER the backbone's initialisation, so that for a given seed
        # the backbone weights are identical with and without the head: the
        # head draws its own random numbers last instead of shifting the
        # backbone's. `None` registers nothing, so the baseline state_dict
        # keeps its historical key set.
        self.depth_head = (
            DepthHead(
                in_channels_l3=256 * block.expansion,
                in_channels_l4=512 * block.expansion,
                width=depth_head_width,
            )
            if depth_head
            else None
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _features(self, x):
        """Stem and the four stages. Returns the layer3 and layer4 maps."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        return f3, f4

    def _classify(self, f4):
        x = self.avgpool(f4)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def forward(self, x):
        """Logits only. The same ops in the same order as before the depth
        head existed; the head is never called here."""
        _, f4 = self._features(x)
        return self._classify(f4)

    def forward_with_depth(self, x):
        """One backbone pass, both heads: (logits, depth map).

        The depth map is (B, 1, H/2, W/2) for an input of (B, 3, H, W).
        Used by the multi-task training pipes and by the depth-consistency
        score; nothing else calls it.

        Raises:
            ValueError: when the model was built without a depth head.
        """
        if self.depth_head is None:
            msg = (
                "forward_with_depth needs a depth head; build the model with "
                "depth_head=True (model=resnet18_depth)."
            )
            raise ValueError(msg)
        f3, f4 = self._features(x)
        return self._classify(f4), self.depth_head(f3, f4)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained_backbone(model, model_urls["resnet18"])
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained_backbone(model, model_urls["resnet34"])
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained_backbone(model, model_urls["resnet50"])
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained_backbone(model, model_urls["resnet101"])
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        _load_pretrained_backbone(model, model_urls["resnet152"])
    return model
