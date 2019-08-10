from deeplab_xception.modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, **kwargs):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm, **kwargs)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, **kwargs)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, **kwargs)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, **kwargs)
    else:
        raise NotImplementedError
