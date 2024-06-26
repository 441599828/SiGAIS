import torch
from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import Conv2d, get_norm


class features_encoder_stem(nn.Module):
    def __init__(self, norm, input_shape=3):
        super().__init__()
        self.conv1 = Conv2d(input_shape,  64, kernel_size=3, stride=2, padding=1, bias=False, norm=get_norm(norm,  64))
        self.conv2 = Conv2d(         64,  64, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm,  64))
        self.conv3 = Conv2d(         64, 128, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, 128))
 
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

class features_encoder_2(nn.Module):
    def __init__(self, norm='BN'):
        super().__init__()
        self.fe2blk1conv1    = Conv2d(128,  64, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  64))
        self.fe2blk1conv2    = Conv2d( 64,  64, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm,  64))
        self.fe2blk1conv3    = Conv2d( 64, 256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 256))
        self.fe2blk1shortcut = Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 256))

        weight_init.c2_msra_fill(self.fe2blk1conv1)
        weight_init.c2_msra_fill(self.fe2blk1conv2)
        weight_init.c2_msra_fill(self.fe2blk1conv3)
        weight_init.c2_msra_fill(self.fe2blk1shortcut)

        self.fe2blk2conv1    = Conv2d(256,  64, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  64))
        self.fe2blk2conv2    = Conv2d( 64,  64, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm,  64))
        self.fe2blk2conv3    = Conv2d( 64, 256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 256))

        weight_init.c2_msra_fill(self.fe2blk2conv1)
        weight_init.c2_msra_fill(self.fe2blk2conv2)
        weight_init.c2_msra_fill(self.fe2blk2conv3)

        self.fe2blk3conv1    = Conv2d(256,  64, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  64))
        self.fe2blk3conv2    = Conv2d( 64,  64, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm,  64))
        self.fe2blk3conv3    = Conv2d( 64, 256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 256))

        weight_init.c2_msra_fill(self.fe2blk3conv1)
        weight_init.c2_msra_fill(self.fe2blk3conv2)
        weight_init.c2_msra_fill(self.fe2blk3conv3)

    def forward(self, x):
        # fe2block1
        blk1out = self.fe2blk1conv1(x)
        blk1out = F.relu(blk1out)
        blk1out = self.fe2blk1conv2(blk1out)
        blk1out = F.relu(blk1out)
        blk1out = self.fe2blk1conv3(blk1out)
        shortcut = self.fe2blk1shortcut(x)
        blk1out = blk1out + shortcut
        blk1out = F.relu(blk1out)
        # fe2block2
        blk2out = self.fe2blk2conv1(blk1out)
        blk2out = F.relu(blk2out)
        blk2out = self.fe2blk2conv2(blk2out)
        blk2out = F.relu(blk2out)
        blk2out = self.fe2blk2conv3(blk2out)
        blk2out = blk2out + blk1out
        blk2out = F.relu(blk2out)
        # fe2block3
        blk3out = self.fe2blk3conv1(blk2out)
        blk3out = F.relu(blk3out)
        blk3out = self.fe2blk3conv2(blk3out)
        blk3out = F.relu(blk3out)
        blk3out = self.fe2blk3conv3(blk3out)
        blk3out = blk3out + blk2out
        blk3out = F.relu(blk3out)
        return blk3out

class features_encoder_3(nn.Module):
    def __init__(self, norm="BN"):
        super().__init__()
        self.fe3blk1conv1    = Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 128))
        self.fe3blk1conv2    = Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False, norm=get_norm(norm, 128))
        self.fe3blk1conv3    = Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 512))
        self.fe3blk1shortcut = Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False, norm=get_norm(norm, 512))

        weight_init.c2_msra_fill(self.fe3blk1conv1)
        weight_init.c2_msra_fill(self.fe3blk1conv2)
        weight_init.c2_msra_fill(self.fe3blk1conv3)
        weight_init.c2_msra_fill(self.fe3blk1shortcut)

        self.fe3blk2conv1    = Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 128))
        self.fe3blk2conv2    = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, 128))
        self.fe3blk2conv3    = Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 512))

        weight_init.c2_msra_fill(self.fe3blk2conv1)
        weight_init.c2_msra_fill(self.fe3blk2conv2)
        weight_init.c2_msra_fill(self.fe3blk2conv3)

        self.fe3blk3conv1    = Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 128))
        self.fe3blk3conv2    = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, 128))
        self.fe3blk3conv3    = Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 512))

        weight_init.c2_msra_fill(self.fe3blk3conv1)
        weight_init.c2_msra_fill(self.fe3blk3conv2)
        weight_init.c2_msra_fill(self.fe3blk3conv3)

        self.fe3blk4conv1    = Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 128))
        self.fe3blk4conv2    = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, 128))
        self.fe3blk4conv3    = Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 512))

        weight_init.c2_msra_fill(self.fe3blk4conv1)
        weight_init.c2_msra_fill(self.fe3blk4conv2)
        weight_init.c2_msra_fill(self.fe3blk4conv3)

    def forward(self, x):
        # res3block1
        blk1out = self.fe3blk1conv1(x)
        blk1out = F.relu(blk1out)
        blk1out = self.fe3blk1conv2(blk1out)
        blk1out = F.relu(blk1out)
        blk1out = self.fe3blk1conv3(blk1out)
        shortcut = self.fe3blk1shortcut(x)
        blk1out = blk1out + shortcut
        blk1out = F.relu(blk1out)
        # res3block2
        blk2out = self.fe3blk2conv1(blk1out)
        blk2out = F.relu(blk2out)
        blk2out = self.fe3blk2conv2(blk2out)
        blk2out = F.relu(blk2out)
        blk2out = self.fe3blk2conv3(blk2out)
        blk2out = blk2out + blk1out
        blk2out = F.relu(blk2out)
        # res3block3
        blk3out = self.fe3blk3conv1(blk2out)
        blk3out = F.relu(blk3out)
        blk3out = self.fe3blk3conv2(blk3out)
        blk3out = F.relu(blk3out)
        blk3out = self.fe3blk3conv3(blk3out)
        blk3out = blk3out + blk2out
        blk3out = F.relu(blk3out)
        # res3block4
        blk4out = self.fe3blk4conv1(blk3out)
        blk4out = F.relu(blk4out)
        blk4out = self.fe3blk4conv2(blk4out)
        blk4out = F.relu(blk4out)
        blk4out = self.fe3blk4conv3(blk4out)
        blk4out = blk4out + blk3out
        blk4out = F.relu(blk4out)
        return blk4out

class features_encoder_4(nn.Module):
    def __init__(self, norm="BN"):
        super().__init__()
        self.fe4blk1conv1    = Conv2d(512,  256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  256))
        self.fe4blk1conv2    = Conv2d(256,  256, kernel_size=3, stride=2, padding=1, bias=False, norm=get_norm(norm,  256))
        self.fe4blk1conv3    = Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 1024))
        self.fe4blk1shortcut = Conv2d(512, 1024, kernel_size=1, stride=2, padding=0, bias=False, norm=get_norm(norm, 1024))

        weight_init.c2_msra_fill(self.fe4blk1conv1)
        weight_init.c2_msra_fill(self.fe4blk1conv2)
        weight_init.c2_msra_fill(self.fe4blk1conv3)
        weight_init.c2_msra_fill(self.fe4blk1shortcut)

        self.fe4blk2conv1    = Conv2d(1024,  256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  256))
        self.fe4blk2conv2    = Conv2d( 256,  256, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm,  256))
        self.fe4blk2conv3    = Conv2d( 256, 1024, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 1024))

        weight_init.c2_msra_fill(self.fe4blk2conv1)
        weight_init.c2_msra_fill(self.fe4blk2conv2)
        weight_init.c2_msra_fill(self.fe4blk2conv3)

        self.fe4blk3conv1    = Conv2d(1024,  256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  256))
        self.fe4blk3conv2    = Conv2d( 256,  256, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm,  256))
        self.fe4blk3conv3    = Conv2d( 256, 1024, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 1024))

        weight_init.c2_msra_fill(self.fe4blk3conv1)
        weight_init.c2_msra_fill(self.fe4blk3conv2)
        weight_init.c2_msra_fill(self.fe4blk3conv3)

        self.fe4blk4conv1    = Conv2d(1024,  256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  256))
        self.fe4blk4conv2    = Conv2d( 256,  256, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm,  256))
        self.fe4blk4conv3    = Conv2d( 256, 1024, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 1024))

        weight_init.c2_msra_fill(self.fe4blk4conv1)
        weight_init.c2_msra_fill(self.fe4blk4conv2)
        weight_init.c2_msra_fill(self.fe4blk4conv3)

        self.fe4blk5conv1    = Conv2d(1024,  256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  256))
        self.fe4blk5conv2    = Conv2d( 256,  256, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm,  256))
        self.fe4blk5conv3    = Conv2d( 256, 1024, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 1024))

        weight_init.c2_msra_fill(self.fe4blk5conv1)
        weight_init.c2_msra_fill(self.fe4blk5conv2)
        weight_init.c2_msra_fill(self.fe4blk5conv3)

        self.fe4blk6conv1    = Conv2d(1024,  256, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  256))
        self.fe4blk6conv2    = Conv2d( 256,  256, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm,  256))
        self.fe4blk6conv3    = Conv2d( 256, 1024, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 1024))

        weight_init.c2_msra_fill(self.fe4blk6conv1)
        weight_init.c2_msra_fill(self.fe4blk6conv2)
        weight_init.c2_msra_fill(self.fe4blk6conv3)

    def forward(self, x):
        # res4block1
        blk1out = self.fe4blk1conv1(x)
        blk1out = F.relu(blk1out)
        blk1out = self.fe4blk1conv2(blk1out)
        blk1out = F.relu(blk1out)
        blk1out = self.fe4blk1conv3(blk1out)
        shortcut = self.fe4blk1shortcut(x)
        blk1out = blk1out + shortcut
        blk1out = F.relu(blk1out)
        # res4block2
        blk2out = self.fe4blk2conv1(blk1out)
        blk2out = F.relu(blk2out)
        blk2out = self.fe4blk2conv2(blk2out)
        blk2out = F.relu(blk2out)
        blk2out = self.fe4blk2conv3(blk2out)
        blk2out = blk2out + blk1out
        blk2out = F.relu(blk2out)
        # res4block3
        blk3out = self.fe4blk3conv1(blk2out)
        blk3out = F.relu(blk3out)
        blk3out = self.fe4blk3conv2(blk3out)
        blk3out = F.relu(blk3out)
        blk3out = self.fe4blk3conv3(blk3out)
        blk3out = blk3out + blk2out
        blk3out = F.relu(blk3out)
        # res4block4
        blk4out = self.fe4blk4conv1(blk3out)
        blk4out = F.relu(blk4out)
        blk4out = self.fe4blk4conv2(blk4out)
        blk4out = F.relu(blk4out)
        blk4out = self.fe4blk4conv3(blk4out)
        blk4out = blk4out + blk3out
        blk4out = F.relu(blk4out)
        # res4block5
        blk5out = self.fe4blk5conv1(blk4out)
        blk5out = F.relu(blk5out)
        blk5out = self.fe4blk5conv2(blk5out)
        blk5out = F.relu(blk5out)
        blk5out = self.fe4blk5conv3(blk5out)
        blk5out = blk5out + blk4out
        blk5out = F.relu(blk5out)
        # res4block6
        blk6out = self.fe4blk6conv1(blk5out)
        blk6out = F.relu(blk6out)
        blk6out = self.fe4blk6conv2(blk6out)
        blk6out = F.relu(blk6out)
        blk6out = self.fe4blk6conv3(blk6out)
        blk6out = blk6out + blk5out
        blk6out = F.relu(blk6out)
        return blk6out

class features_encoder_5(nn.Module):
    def __init__(self, norm="BN"):
        super().__init__()
        self.fe5blk1conv1    = Conv2d(1024,  512, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  512))
        self.fe5blk1conv2    = Conv2d( 512,  512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False, norm=get_norm(norm,  512))
        self.fe5blk1conv3    = Conv2d( 512, 2048, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 2048))
        self.fe5blk1shortcut = Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 2048))

        weight_init.c2_msra_fill(self.fe5blk1conv1)
        weight_init.c2_msra_fill(self.fe5blk1conv2)
        weight_init.c2_msra_fill(self.fe5blk1conv3)
        weight_init.c2_msra_fill(self.fe5blk1shortcut)

        self.fe5blk2conv1    = Conv2d(2048,  512, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  512))
        self.fe5blk2conv2    = Conv2d( 512,  512, kernel_size=3, stride=1, padding=4, dilation=4 ,bias=False, norm=get_norm(norm,  512))
        self.fe5blk2conv3    = Conv2d( 512, 2048, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 2048))

        weight_init.c2_msra_fill(self.fe5blk2conv1)
        weight_init.c2_msra_fill(self.fe5blk2conv2)
        weight_init.c2_msra_fill(self.fe5blk2conv3)

        self.fe5blk3conv1    = Conv2d(2048,  512, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm,  512))
        self.fe5blk3conv2    = Conv2d( 512,  512, kernel_size=3, stride=1, padding=8, dilation=8, bias=False, norm=get_norm(norm,  512))
        self.fe5blk3conv3    = Conv2d( 512, 2048, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm(norm, 2048))

        weight_init.c2_msra_fill(self.fe5blk3conv1)
        weight_init.c2_msra_fill(self.fe5blk3conv2)
        weight_init.c2_msra_fill(self.fe5blk3conv3)

    def forward(self, x):
        # res5block1
        blk1out = self.fe5blk1conv1(x)
        blk1out = F.relu(blk1out)
        blk1out = self.fe5blk1conv2(blk1out)
        blk1out = F.relu(blk1out)
        blk1out = self.fe5blk1conv3(blk1out)
        shortcut = self.fe5blk1shortcut(x)
        blk1out = blk1out + shortcut
        blk1out = F.relu(blk1out)
        # res5block2
        blk2out = self.fe5blk2conv1(blk1out)
        blk2out = F.relu(blk2out)
        blk2out = self.fe5blk2conv2(blk2out)
        blk2out = F.relu(blk2out)
        blk2out = self.fe5blk2conv3(blk2out)
        blk2out = blk2out + blk1out
        blk2out = F.relu(blk2out)
        # res5block3
        blk3out = self.fe5blk3conv1(blk2out)
        blk3out = F.relu(blk3out)
        blk3out = self.fe5blk3conv2(blk3out)
        blk3out = F.relu(blk3out)
        blk3out = self.fe5blk3conv3(blk3out)
        blk3out = blk3out + blk2out
        blk3out = F.relu(blk3out)
        return blk3out

class sigais_backbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        norm = cfg.MODEL.RESNETS.NORM


        self.features_encoder_1 = features_encoder_stem(norm, input_shape)
        self.features_encoder_2 = features_encoder_2(norm)
        self.features_encoder_3 = features_encoder_3(norm)
        self.features_encoder_4 = features_encoder_4(norm)
        self.features_encoder_5 = features_encoder_5(norm)

    def forward(self, img, bkg):
        out_features = {}

        [img, bkg] = [self.features_encoder_1(img), self.features_encoder_1(bkg)]
        [img, bkg] = [self.features_encoder_2(img), self.features_encoder_2(bkg)]
        out_features['fe2'] = {}
        out_features['fe2']['img'] = img
        out_features['fe2']['bkg'] = bkg
        [img, bkg] = [self.features_encoder_3(img), self.features_encoder_3(bkg)]
        out_features['fe3'] = {}
        out_features['fe3']['img'] = img
        out_features['fe3']['bkg'] = bkg
        [img, bkg] = [self.features_encoder_4(img), self.features_encoder_4(bkg)]
        [img, bkg] = [self.features_encoder_5(img), self.features_encoder_5(bkg)]
        out_features['fe5'] = {}
        out_features['fe5']['img'] = img
        out_features['fe5']['bkg'] = bkg          

        return out_features


@BACKBONE_REGISTRY.register()
def build_sigais_backbone(cfg, input_shape):
    """
    Hard code implement of resnet 52 with depthwise separable conv
    """
    return sigais_backbone(cfg, input_shape.channels)
