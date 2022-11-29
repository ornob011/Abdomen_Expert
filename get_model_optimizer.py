import torch
import timm
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
import torchvision

DEVICE = 'cuda'

feature_extract = True
LR = 0.0001


def get_model():

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    class SwinTransformer(nn.Module):
        def __init__(self, emb_size=512):
            super(SwinTransformer, self).__init__()
            self.network = timm.create_model(
                'swin_base_patch4_window7_224_in22k', pretrained=True, progress=True)
            set_parameter_requires_grad(self.network, feature_extract)

            self.network.head = nn.Linear(
                self.network.head.in_features, out_features=emb_size)

        def forward(self, images):
            embeddings = self.network(images)
            return embeddings

    class MobileNetV2(nn.Module):
        def __init__(self, emb_size=512):
            super(MobileNetV2, self).__init__()
            self.network = models.mobilenet_v2(pretrained=True)
            set_parameter_requires_grad(self.network, feature_extract)

            self.network.classifier[1] = nn.Linear(
                self.network.classifier[1].in_features, out_features=emb_size)

        def forward(self, images):
            embeddings = self.network(images)
            return embeddings

    class EfficientNetB0(nn.Module):
        def __init__(self, emb_size=512):
            super(EfficientNetB0, self).__init__()
            self.network = models.efficientnet_b0(pretrained=True)
            set_parameter_requires_grad(self.network, feature_extract)

            self.network.classifier[1] = nn.Linear(
                in_features=self.network.classifier[1].in_features, out_features=emb_size)

        def forward(self, images):
            embeddings = self.network(images)
            return embeddings

    class Resnext101_32x8d(nn.Module):
        def __init__(self, emb_size=512):
            super(Resnext101_32x8d, self).__init__()
            self.network = models.resnext101_32x8d(
                pretrained=True, progress=True)
            set_parameter_requires_grad(self.network, feature_extract)

            self.network.fc = nn.Linear(
                in_features=self.network.fc.in_features, out_features=emb_size)

        def forward(self, images):
            embeddings = self.network(images)
            return embeddings

    class VGG16(nn.Module):
        def __init__(self, emb_size=512):
            super(VGG16, self).__init__()
            self.network = models.vgg16(pretrained=True, progress=True)
            set_parameter_requires_grad(self.network, feature_extract)

            self.network.classifier[6] = nn.Linear(
                in_features=self.network.classifier[6].in_features, out_features=emb_size)

        def forward(self, images):
            embeddings = self.network(images)
            return embeddings

    class Bit(nn.Module):
        def __init__(self, emb_size=512):
            super(Bit, self).__init__()
            self.network = timm.create_model(
                'resnetv2_101x1_bitm', pretrained=True)
            set_parameter_requires_grad(self.network, feature_extract)
            self.network.head.fc = nn.Conv2d(
                2048, 512, kernel_size=(1, 1), stride=(1, 1))

        def forward(self, images):
            embeddings = self.network(images)
            return embeddings

    class Ensemble(nn.Module):
        def __init__(self, Bit, VGG16, SwinTransformer, MobileNetV2, EfficientNetB0, Resnext101_32x8d):
            super(Ensemble, self).__init__()

            self.Bit = Bit
            self.VGG16 = VGG16
            self.SwinTransformer = SwinTransformer
            self.MobileNetV2 = MobileNetV2
            self.EfficientNetB0 = EfficientNetB0
            self.Resnext101_32x8d = Resnext101_32x8d
            self.classifier = nn.Linear(512*6, 512)

        def forward(self, y):
            x0 = self.Bit(y)
            x1 = self.VGG16(y)
            x2 = self.SwinTransformer(y)
            x4 = self.MobileNetV2(y)
            x5 = self.EfficientNetB0(y)
            x8 = self.Resnext101_32x8d(y)
            x = torch.cat((x0, x1, x2, x4, x5, x8), dim=1)
            x = self.classifier(F.relu(x))
            return x

    VGG16 = VGG16().to(DEVICE)
    SwinTransformer = SwinTransformer().to(DEVICE)
    MobileNetV2 = MobileNetV2().to(DEVICE)
    EfficientNetB0 = EfficientNetB0().to(DEVICE)
    Resnext101_32x8d = Resnext101_32x8d().to(DEVICE)
    Bit = Bit()

    model = Ensemble(Bit, VGG16, SwinTransformer, MobileNetV2,
                     EfficientNetB0, Resnext101_32x8d)

    model.to(DEVICE)

    return model


def get_optimizer():
    model = get_model()
    params_to_update = model.parameters()
    
    print("Params to learn:")
    
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    optimizer = torch.optim.Adam(params_to_update, weight_decay=1e-5, lr=LR)

    return optimizer
