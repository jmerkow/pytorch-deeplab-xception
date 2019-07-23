import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplab_xception.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from deeplab_xception.modeling.aspp import build_aspp
from deeplab_xception.modeling.decoder import build_decoder
from deeplab_xception.modeling.backbone import build_backbone


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class DeepLab(nn.Module):
    def __init__(self, encoder='resnet', output_stride=16, classes=21,
                 sync_bn=True, freeze_bn=False, model_dir=None, activation='sigmoid', encoder_classify=False,
                 **kwargs):
        print(kwargs)

        super(DeepLab, self).__init__()
        self.name = 'deeplab-{}'.format(encoder)
        if encoder == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if encoder == 'drn':
            inplanes = 512
        elif encoder == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048

        self.backbone = build_backbone(encoder, output_stride, BatchNorm, model_dir=model_dir)
        self.aspp = build_aspp(encoder, output_stride, BatchNorm)
        self.decoder = build_decoder(classes, encoder, BatchNorm)

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

        self.encoder_classifier = None
        self.classes = classes

        self.encoder_classify = encoder_classify
        if self.encoder_classify:
            self.encoder_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(inplanes, self.classes),
            )
            self.name += "-wclassifier"

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        features, low_level_feat = self.backbone(input)
        mask = self.aspp(features)
        mask = self.decoder(mask, low_level_feat)
        mask = F.interpolate(mask, size=input.size()[2:], mode='bilinear', align_corners=True)

        if self.encoder_classify:
            score = self.encoder_classifier(features)
            return [mask, score]
        return mask

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            mask = self.forward(x)
            score = None
            if self.encoder_classify:
                mask, score = mask
            if self.activation:
                mask = self.activation(mask)
                if score:
                    score = self.activation(score)
            if self.encoder_classify:
                x = [mask, score]
            else:
                x = mask

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_encoder_params(self):
        return self.get_1x_lr_params()

    def get_decoder_params(self):
        return self.get_10x_lr_params()

    def get_classifier_params(self):
        return self.encoder_classifier.parameters()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(encoder='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


