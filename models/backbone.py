import timm

swin = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained = True,
        features_only = True,
        out_indices = (-1,) # c5 final layer
        )
swin.eval()
