import torch
from torch import nn
import torch.nn.functional as F
from detectron2.utils.registry import Registry
from detectron2.layers import ASPP, Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from .loss import DeepLabCE
import fvcore.nn.weight_init as weight_init

DECODER_REGISTRY = Registry("DECODER")
DECODER_REGISTRY.__doc__ = """
Registry for semantic segmentation heads and ins head.
"""


class aspp(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.aspp = ASPP(in_channels=2048, out_channels=256, dilations=[6,12,18], norm=norm,
                        activation=F.relu, pool_kernel_size=(34, 60), dropout=0.1, use_depthwise_separable_conv=True)
    
    def forward(self, x):
        x = self.aspp(x)
        return x

class sem_seg_predictor_no_pan(nn.Module):
    def __init__(self, norm):
        super().__init__()

        self.seg_head = DepthwiseSeparableConv2d(256,256,kernel_size=5,padding=2,norm1=norm,activation1=F.relu,norm2=norm,activation2=F.relu)
        self.seg_predictor = Conv2d(256, 5, kernel_size=1)
        nn.init.normal_(self.seg_predictor.weight, 0, 0.001)
        nn.init.constant_(self.seg_predictor.bias, 0)

    def forward(self, img_seg):
        img_seg = self.seg_head(img_seg)
        img_seg = self.seg_predictor(img_seg)
        return img_seg


class center_predictor_pan(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.center_head = nn.Sequential(
            Conv2d(384, 128, kernel_size=3, padding=1, bias=False, norm=get_norm(norm, 128), activation=F.relu),
            Conv2d(128,  32, kernel_size=3, padding=1, bias=False, norm=get_norm(norm,  32), activation=F.relu)
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(32, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

    def forward(self, img_ins):
        center = self.center_head(img_ins)
        center = self.center_predictor(center)
        return center


class offset_predictor_pan(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.offset_head = DepthwiseSeparableConv2d(384,32,kernel_size=5,padding=2,norm1=norm,activation1=F.relu,norm2=norm,activation2=F.relu)
        self.offset_predictor = Conv2d(32, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

    def forward(self, img_ins):
        offset = self.offset_head(img_ins)
        offset = self.offset_predictor(offset)
        return offset

#Difference module
class conv_diff_article(nn.Module):
    def __init__(self, norm, in_channels, out_channels):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_max = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_avg = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        )

        self.conv = Conv2d(2, 1, kernel_size=3, stride=1, padding=1, norm=get_norm(norm, 1))

        self.conv1 = Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, norm=get_norm(norm, out_channels))
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=get_norm(norm, out_channels))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, img, bkg):
        avg_minus_1 = torch.abs(self.avg_pool(img) - self.avg_pool(bkg))
        max_minus_1 = torch.abs(self.max_pool(img) - self.max_pool(bkg))
        sumup_1 = self.sigmoid(self.mlp_avg(avg_minus_1) + self.mlp_max(max_minus_1))
        img = img * sumup_1
        bkg = bkg * sumup_1

        img_avg = torch.mean(img, dim=1, keepdim=True)
        img_max, _ = torch.max(img, dim=1, keepdim=True)
        bkg_avg = torch.mean(bkg, dim=1, keepdim=True)
        bkg_max, _ = torch.max(bkg, dim=1, keepdim=True)

        avg_minus_2 = torch.abs(img_avg - bkg_avg)
        max_minus_2 = torch.abs(img_max - bkg_max)
        sumup_2 = self.sigmoid(self.conv(torch.concat((avg_minus_2, max_minus_2),dim=1)))

        img = img * sumup_2
        bkg = bkg * sumup_2

        img = torch.concat((img, bkg), dim=1)
        img = self.relu(self.conv2(self.relu(self.conv1(img))))
        return img


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

#Intermediate prediction module
class make_prediction(nn.Module):
    def __init__(self, norm, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=get_norm(norm, out_channels))
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=get_norm(norm, out_channels))
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        self.relu = nn.ReLU()
    def forward(self, img):
        img = self.relu(self.conv1(img))
        img = self.conv2(img)
        return img


class sigais_decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        norm = cfg.MODEL.DECODER.NORM

        # front mask branch
        self.conv_diff_2 = conv_diff_article(norm, 256, 256)
        self.conv_diff_3 = conv_diff_article(norm, 512, 256)
        self.conv_diff_5 = conv_diff_article(norm, 2048, 256)

        self.make_pred_2 = make_prediction(norm, 256, 2)
        self.make_pred_3 = make_prediction(norm, 256, 2)
        self.make_pred_5 = make_prediction(norm, 256, 2)

        self.linear_fuse = Conv2d(768, 256, kernel_size=1, padding=0, stride=1, norm=get_norm(norm, 256))
        weight_init.c2_msra_fill(self.linear_fuse)
        self.relu = nn.ReLU()
        self.dense_1 = ResidualBlock(256)
        self.fm_predictor = Conv2d(256, 2,kernel_size=3, stride=1, padding=1, norm=get_norm(norm, 2))
        self.sigmoid = nn.Sigmoid()
        nn.init.normal_(self.fm_predictor.weight, 0, 0.001)
        nn.init.constant_(self.fm_predictor.bias, 0)
        self.fm_loss = nn.CrossEntropyLoss()

        # sem_seg_branch
        self.seg_aspp = aspp(norm)

        self.seg_proj_conv_1 = Conv2d(512,  64, kernel_size=1, bias=False, norm=get_norm(norm,  64), activation=F.relu)
        weight_init.c2_msra_fill(self.seg_proj_conv_1)
        self.seg_fuse_conv_1 = DepthwiseSeparableConv2d(in_channels=320,out_channels=256,kernel_size=5,padding=2,norm1=norm,activation1=F.relu,norm2=norm,activation2=F.relu,)

        self.seg_proj_conv_2 = Conv2d(256,  32, kernel_size=1, bias=False, norm=get_norm(norm,  32), activation=F.relu)
        weight_init.c2_msra_fill(self.seg_proj_conv_2)
        self.seg_fuse_conv_2 = DepthwiseSeparableConv2d(in_channels=288,out_channels=256,kernel_size=5,padding=2,norm1=norm,activation1=F.relu,norm2=norm,activation2=F.relu,)

        self.sem_seg_predictor = sem_seg_predictor_no_pan(norm)

        # ins_branch
        self.ins_aspp = aspp(norm)
        
        self.ins_proj_conv_1 = Conv2d(512,  64, kernel_size=1, bias=False, norm=get_norm(norm,  64), activation=F.relu)
        weight_init.c2_msra_fill(self.ins_proj_conv_1)
        self.ins_fuse_conv_1 = DepthwiseSeparableConv2d(in_channels=320,out_channels=128,kernel_size=5,padding=2,norm1=norm,activation1=F.relu,norm2=norm,activation2=F.relu,)

        self.ins_proj_conv_2 = Conv2d(256,  32, kernel_size=1, bias=False, norm=get_norm(norm,  32), activation=F.relu)
        weight_init.c2_msra_fill(self.ins_proj_conv_2)
        self.ins_fuse_conv_2 = DepthwiseSeparableConv2d(in_channels=160,out_channels=128,kernel_size=5,padding=2,norm1=norm,activation1=F.relu,norm2=norm,activation2=F.relu,)
        
        # if self.inspan:
        self.ins_proj_conv_3 = Conv2d(128, 16, kernel_size=1, bias=False, norm=get_norm(norm,16), activation=F.relu)
        weight_init.c2_msra_fill(self.ins_proj_conv_3)
        self.ins_fuse_conv_3 = DepthwiseSeparableConv2d(in_channels=144, out_channels=128, kernel_size=5,padding=2,norm1=norm,activation1=F.relu,norm2=norm,activation2=F.relu,)
        
        self.ins_proj_conv_4 = Conv2d(256, 16, kernel_size=1, bias=False, norm=get_norm(norm,16), activation=F.relu)
        weight_init.c2_msra_fill(self.ins_proj_conv_4)
        self.ins_fuse_conv_4 = DepthwiseSeparableConv2d(in_channels=144, out_channels=128, kernel_size=5,padding=2,norm1=norm,activation1=F.relu,norm2=norm,activation2=F.relu,)
        self.center_predictor = center_predictor_pan(norm)
        self.offset_predictor = offset_predictor_pan(norm)

        # losses
        self.seg_loss = DeepLabCE(ignore_label=255, top_k_percent_pixels=0.2)
        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    def forward(self, features,  fm_targets=None, seg_targets=None, seg_weights=None,\
                center_targets=None, center_weights=None, offset_targets=None, offset_weights=None):
        outputs = {}

        front_mask_2 = self.conv_diff_2(features['fe2']['img'], features['fe2']['bkg'])
        front_mask_3 = self.conv_diff_3(features['fe3']['img'], features['fe3']['bkg'])
        front_mask_5 = self.conv_diff_5(features['fe5']['img'], features['fe5']['bkg'])

        if self.training:
            outputs['fm_middle_2'] = self.make_pred_2(front_mask_2)
            outputs['fm_middle_3'] = self.make_pred_3(front_mask_3)
            outputs['fm_middle_5'] = self.make_pred_5(front_mask_5)

        front_mask_3 = F.interpolate(front_mask_3, size=front_mask_2.shape[2:], mode='bilinear', align_corners=False)
        front_mask_5 = F.interpolate(front_mask_5, size=front_mask_2.shape[2:], mode='bilinear', align_corners=False)
        front_mask = torch.cat((front_mask_2, front_mask_3, front_mask_5), dim=1)
        front_mask = self.linear_fuse(front_mask)
        front_mask = self.relu(front_mask)

        front_mask = self.dense_1(front_mask)
        front_mask = self.fm_predictor(front_mask)
        
        outputs['fm'] = front_mask
        front_mask = torch.argmax(front_mask, dim=1, keepdim=True).float()

        features['fe5']['img'] = features['fe5']['img'] * F.interpolate(front_mask, size=features['fe5']['img'].shape[2:], mode='bilinear', align_corners=False)
        features['fe3']['img'] = features['fe3']['img'] * F.interpolate(front_mask, size=features['fe3']['img'].shape[2:], mode='bilinear', align_corners=False)
        features['fe2']['img'] = features['fe2']['img'] * F.interpolate(front_mask, size=features['fe2']['img'].shape[2:], mode='bilinear', align_corners=False)

        # sem_seg_branch
        img_seg_1 = self.seg_aspp(features['fe5']['img'])

        img_seg_backbone_1 = self.seg_proj_conv_1(features['fe3']['img'])
        img_seg_2 = self.seg_fuse_conv_1(
            torch.cat((F.interpolate(img_seg_1, size=img_seg_backbone_1.shape[2:],mode='bilinear',align_corners=False), img_seg_backbone_1), dim=1)
        )

        img_seg_backbone_2 = self.seg_proj_conv_2(features['fe2']['img'])
        img_seg_3 = self.seg_fuse_conv_2(
            torch.cat((F.interpolate(img_seg_2, size=img_seg_backbone_2.shape[2:], mode='bilinear', align_corners=False), img_seg_backbone_2), dim=1)
        )

        sem_seg = self.sem_seg_predictor(img_seg_3)
            
        outputs['seg'] = sem_seg

        # ins_branch
        img_ins_1 = self.ins_aspp(features['fe5']['img'])
        
        img_ins_backbone_1 = self.ins_proj_conv_1(features['fe3']['img'])
        img_ins_2 = self.ins_fuse_conv_1(
            torch.cat((F.interpolate(img_ins_1, size=img_ins_backbone_1.shape[2:], mode='bilinear', align_corners=False), img_ins_backbone_1), dim=1)
            )
        
        img_ins_backbone_2 = self.ins_proj_conv_2(features['fe2']['img'])
        img_ins_3 = self.ins_fuse_conv_2(
            torch.cat((F.interpolate(img_ins_2, img_ins_backbone_2.shape[2:], mode='bilinear', align_corners=False), img_ins_backbone_2), dim=1)
            )
        
        # if self.inspan:
        img_ins_4 = self.ins_proj_conv_3(img_ins_2)
        img_ins_5 = self.ins_fuse_conv_3(
            torch.cat((F.interpolate(img_ins_3, size=img_ins_4.shape[2:], mode='bilinear', align_corners=False), img_ins_4), dim=1)
        )

        img_ins_6 = self.ins_proj_conv_4(img_ins_1)
        img_ins_7 = self.ins_fuse_conv_4(
            torch.cat((F.interpolate(img_ins_5, size=img_ins_6.shape[2:], mode="bilinear", align_corners=False), img_ins_6), dim=1)
        )
        img_ins = torch.cat((
            img_ins_3,
            F.interpolate(img_ins_5, size=img_ins_3.shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(img_ins_7, size=img_ins_3.shape[2:], mode='bilinear', align_corners=False),
            ), dim=1)
        center = self.center_predictor(img_ins)
        offset = self.offset_predictor(img_ins)

        outputs['center'] = center
        outputs['offset'] = offset


        # training losses with weights
        if self.training:
            fm_loss = self.fm_losses(outputs['fm'], outputs['fm_middle_2'], outputs['fm_middle_3'], outputs['fm_middle_5'], fm_targets)

            fm_losses = {"loss_fm": fm_loss * 5.}
            seg_img_loss = self.seg_losses(outputs['seg'], seg_targets, seg_weights)
            seg_losses = {"loss_sem_seg": seg_img_loss * 1.}
            center_img_loss = self.center_losses(outputs['center'], center_targets, center_weights)
            center_losses = {"loss_center": center_img_loss * 300.}
            offset_img_loss = self.offset_losses(outputs['offset'], offset_targets, offset_weights)
            offset_losses = {"loss_offset": offset_img_loss * 0.02}

            return(None,None,None,None,fm_losses,seg_losses,center_losses,offset_losses)
        else:
            outputs['fm'] = torch.argmax(outputs['fm'], dim=1, keepdim=True).float()
            fm_out = F.interpolate(outputs['fm'], scale_factor=4, mode='bilinear', align_corners=False)

            seg_out = F.interpolate(outputs['seg'], scale_factor=4, mode="bilinear", align_corners=False)
            center_out = F.interpolate(outputs['center'], scale_factor=4, mode="bilinear", align_corners=False)
            offset_out = F.interpolate(outputs['offset'], scale_factor=4, mode="bilinear", align_corners=False) * 4
            return (fm_out,seg_out,center_out,offset_out,{},{},{},{},)

    def fm_losses(self, outputs, outputs_2, outputs_3, outputs_5, fm_targets):
        # fm_loss
        fm_targets_labels = fm_targets.squeeze(1).type(torch.LongTensor).to(outputs.device)
        fm_loss_2 = self.fm_loss(F.interpolate(outputs_2, size=fm_targets.shape[2:], mode="bilinear", align_corners=False), fm_targets_labels)
        fm_loss_3 = self.fm_loss(F.interpolate(outputs_3, size=fm_targets.shape[2:], mode="bilinear", align_corners=False), fm_targets_labels)
        fm_loss_5 = self.fm_loss(F.interpolate(outputs_5, size=fm_targets.shape[2:], mode="bilinear", align_corners=False), fm_targets_labels)
        fm_loss = self.fm_loss(F.interpolate(outputs, size=fm_targets.shape[2:], mode="bilinear", align_corners=False), fm_targets_labels)
        fm_loss = 0.5*fm_loss_2 + 0.5*fm_loss_3 + 0.8*fm_loss_5 + fm_loss
        return fm_loss

    def seg_losses(self, seg_predictions, seg_targets, seg_weights):
        seg_predictions = F.interpolate(seg_predictions, size=seg_targets.shape[1:], mode="bilinear", align_corners=False)
        loss = self.seg_loss(seg_predictions, seg_targets, seg_weights)
        return loss

    def center_losses(self, center_predictions, center_targets, center_weights):
        center_predictions = F.interpolate(center_predictions, size=center_targets.shape[2:],mode="bilinear",align_corners=False)
        loss = self.center_loss(center_predictions, center_targets) * center_weights
        if center_weights.sum() > 0:
            loss = loss.sum() / center_weights.sum()
        else:
            loss = loss.sum() * 0
        return loss

    def offset_losses(self, offset_predictions, offset_targets, offset_weights):
        offset_predictions = F.interpolate(offset_predictions, size=offset_targets.shape[2:], mode="bilinear", align_corners=False) * 4
        loss = self.offset_loss(offset_predictions, offset_targets) * offset_weights
        if offset_weights.sum() > 0:
            loss = loss.sum() / offset_weights.sum()
        else:
            loss = loss.sum() * 0
        return loss


@DECODER_REGISTRY.register()
def build_decoder(cfg):
    # """
    # Hard code implement of resnet 52 with depthwise separable conv
    # """
    name = cfg.MODEL.DECODER.NAME
    if 'build_decoder' not in name:
        NotImplementedError
    return sigais_decoder(cfg)