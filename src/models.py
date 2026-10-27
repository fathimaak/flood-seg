import segmentation_models_pytorch as smp

ARCHS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(
    arch="unet",
    encoder="resnet34",
    encoder_weights="imagenet",
    in_channels=6,
    classes=2,
):
    return ARCHS[arch](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
