import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.DWT_2D import DWT_2D
from models.DWT_3D import DWT_3D
# from timm import create_model

def seq2img(x):
    [b, c, h, w] = x.shape
    d = h * w
    x = x.view(b, c, d)
    x = x.view(b, c, h, w)
    return x

class Spatial_Attn_2d(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attn_2d, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        y = self.attn(y)
        return x * y.expand_as(x)



class Spatial_Spectral_Attn_3d(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Spectral_Attn_3d, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        y = self.attn(y)
        return x * y.expand_as(x)


class LiDAR_Encoder(nn.Module):
    def __init__(self, wavename, in_channels=1, out_channels=64, attn_kernel_size=7):
        super(LiDAR_Encoder, self).__init__()
        self.DWT_layer_2D = DWT_2D(wavename=wavename)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.S_attn = Spatial_Attn_2d(kernel_size=attn_kernel_size)
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, lidar_img):
        x_dwt = self.DWT_layer_2D(lidar_img)

        x_ll = x_dwt[0]
        x_ll = self.conv1(x_ll)
        x_ll = self.conv2(x_ll)
        x_ll = self.S_attn(x_ll)

        x_high = torch.cat([x for x in x_dwt[1:4]], dim=1)
        x_high = self.conv_high(x_high)

        x = torch.cat([x_ll, x_high], dim=1)
        x = self.conv2d(x)
        return x
#------------------------------------------------------------------------------------------
class HSI_Encoder_3D(nn.Module):
    def __init__(self, in_depth, patch_size, wavename,
                 in_channels_3d=1, out_channels_3d=16, out_channels_2d=64, attn_kernel_size=7):
        super(HSI_Encoder_3D, self).__init__()
        self.in_depth = in_depth
        self.patch_size = patch_size

        self.DWT_layer_3D = DWT_3D(wavename=wavename)

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels_3d, out_channels=out_channels_3d // 2, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels_3d // 2),
            nn.ReLU(),
        )

        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels_3d // 2, out_channels=out_channels_3d, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels_3d),
            nn.ReLU(),
        )

        self.SS_attn = Spatial_Spectral_Attn_3d(kernel_size=attn_kernel_size)

        self.conv3d_high = nn.Sequential(
            nn.Conv3d(in_channels=in_channels_3d * 7, out_channels=out_channels_3d, kernel_size=1),
            nn.BatchNorm3d(out_channels_3d),
            nn.ReLU(),
        )

        self.in_channels_2d = int(self.get_inchannels_2d())
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels_2d, out_channels=out_channels_2d, kernel_size=1),
            nn.BatchNorm2d(out_channels_2d),
            nn.ReLU(),
        )


    def get_inchannels_2d(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.in_depth // 2, self.patch_size // 2, self.patch_size // 2))
            x = self.conv3d_1(x)
            x = self.conv3d_2(x)

            x = torch.cat([x, x], dim=1)
            _, t, c, _, _ = x.size()
        return t * c

    def forward(self, hsi_img):

        hsi_img = hsi_img.unsqueeze(1)
        x_dwt = self.DWT_layer_3D(hsi_img.permute(0, 1, 3, 2, 4))
        x_lll = x_dwt[0].permute(0, 1, 3, 2, 4)
        x_lll = self.conv3d_1(x_lll)
        x_lll = self.conv3d_2(x_lll)
        x_lll = self.SS_attn(x_lll)


        x_high = torch.cat([x.permute(0, 1, 3, 2, 4) for x in x_dwt[1:8]], dim=1)
        x_high = self.conv3d_high(x_high)

        x = torch.cat([x_lll, x_high], dim=1)

        x = rearrange(x, 'b c d h w ->b (c d) h w')
        x = self.conv2d(x)

        return x

class HSI_Encoder_2D(nn.Module):
    def __init__(self, wavename, in_channels, out_channels=64, attn_kernel_size=7):
        super(HSI_Encoder_2D, self).__init__()
        self.DWT_layer_2D = DWT_2D(wavename=wavename)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.S_attn = Spatial_Attn_2d(kernel_size=attn_kernel_size)
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, hsi_img):
        x_dwt = self.DWT_layer_2D(hsi_img)

        x_ll = x_dwt[0]
        x_ll = self.conv1(x_ll)
        x_ll = self.conv2(x_ll)
        x_ll = self.S_attn(x_ll)

        x_high = torch.cat([x for x in x_dwt[1:4]], dim=1)
        x_high = self.conv_high(x_high)

        x = torch.cat([x_ll, x_high], dim=1)
        x = self.conv2d(x)
        return x
#-----------------------------------------------------------------------------
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, hsi, lidar):
        query = self.query_proj(hsi)
        key = self.key_proj(lidar)
        value = self.value_proj(lidar)

        attn_output, _ = self.attention(query, key, value)
        gate_weight = self.gate(attn_output)
        output = self.norm(gate_weight * attn_output + (1 - gate_weight) * hsi)
        return output


#--------------------------------------------------------------------------------
class Fusion_Unit(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Fusion_Unit, self).__init__()

        self.conv_1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.BN_1 = nn.BatchNorm2d(dim_out)

        self.deconv = nn.ConvTranspose2d(dim_out, dim_in, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(dim_in)

        self.conv_2 = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.BN_3 = nn.BatchNorm2d(dim_out)

    def forward(self, F_M):

        x_sq = F.gelu(self.BN_1(self.conv_1(F_M)))

        x_ex = F.gelu(self.BN_2(self.deconv(x_sq)))
        if x_ex.size() != F_M.size():
            x_ex = F.interpolate(x_ex, size=F_M.shape[2:], mode='bilinear', align_corners=True)

        residual = F_M - x_ex
        x_r = F.gelu(self.BN_3(self.conv_2(residual)))
        x_out = x_sq + x_r
        return x_out


class Cross_Fusion(nn.Module):
    def __init__(self, dim_head, heads, cls=None):
        super(Cross_Fusion, self).__init__()

        self.convH = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.BN_H1 = nn.BatchNorm2d(128)

        self.convL = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.BN_L1 = nn.BatchNorm2d(128)

        self.num_heads = heads
        self.dim_head = dim_head

        self.Hto_q = nn.Linear(128, dim_head * heads, bias=False)
        self.Hto_k = nn.Linear(128, dim_head * heads, bias=False)
        self.Hto_v = nn.Linear(128, dim_head * heads, bias=False)
        self.Lto_q = nn.Linear(128, dim_head * heads, bias=False)
        self.Lto_k = nn.Linear(128, dim_head * heads, bias=False)
        self.Lto_v = nn.Linear(128, dim_head * heads, bias=False)

        self.rescaleH = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescaleL = nn.Parameter(torch.ones(heads, 1, 1))

        self.projH = nn.Linear(dim_head * heads, 128)
        self.projL = nn.Linear(dim_head * heads, 128)

        self.LN_H2 = nn.BatchNorm2d(128)
        self.LN_L2 = nn.BatchNorm2d(128)

        self.FU_1 = Fusion_Unit(256, 360)
        self.FU_2 = Fusion_Unit(360, 512)
        self.FU_3 = Fusion_Unit(512, 512)

        self.global_block = TinyMLPMixer(512)

    def forward(self, F_H, F_L):
        assert F_H.shape == F_L.shape

        F_H = F.relu(self.BN_H1(self.convH(F_H)))
        F_L = F.relu(self.BN_L1(self.convL(F_L)))

        F_M = torch.cat([F_H, F_L], dim=1)
        F_M = self.FU_1(F_M)
        F_M = self.FU_2(F_M)
        F_M = self.FU_3(F_M)

        F_M = self.global_block(F_M)

        return F_M
#---------------------------------------------------------------------------------
class TinyMLPMixer(nn.Module):
    def __init__(self, in_channels, token_dim=128, channel_dim=512, dropout_rate=0.3):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.token_mlp = nn.Sequential(
            nn.Conv2d(in_channels, token_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(token_dim, in_channels, kernel_size=1)
        )
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels, channel_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(channel_dim, in_channels, kernel_size=1)
        )

    def forward(self, x):

        x_norm = self.norm(x)
        token_out = self.token_mlp(x_norm)
        out = x + self.dropout(token_out)
        channel_out = self.channel_mlp(out)
        out = out + self.dropout(channel_out)
        return out

#---------------------------------------------------------------------------------
class CMFDNET(nn.Module):
    def __init__(self, l1, l2, patch_size, num_classes, wavename,
                 attn_kernel_size, dim_head, heads, embed_dim, num_heads):
        super().__init__()

        self.hsi_encoder_3d = HSI_Encoder_3D(in_depth=l1, patch_size=patch_size, wavename=wavename,
                                             out_channels_2d=128, attn_kernel_size=attn_kernel_size)
        self.hsi_encoder_2d = HSI_Encoder_2D(wavename=wavename, in_channels=l1, out_channels=128,
                                             attn_kernel_size=attn_kernel_size)
        self.lidar_encoder = LiDAR_Encoder(wavename=wavename, in_channels=l2, out_channels=128,
                                           attn_kernel_size=attn_kernel_size)

        self.cross_modal_attn = CrossModalAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.cross_fusion = Cross_Fusion(dim_head=dim_head, heads=heads, cls=num_classes)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, img_hsi, img_lidar):
        x_hsi_3d = self.hsi_encoder_3d(img_hsi)
        x_hsi_2d = self.hsi_encoder_2d(img_hsi)
        x_hsi = x_hsi_3d + x_hsi_2d  # 融合 HSI 特征

        x_lidar = self.lidar_encoder(img_lidar)

        if x_hsi.size(2) != x_lidar.size(2) or x_hsi.size(3) != x_lidar.size(3):
            x_hsi = F.interpolate(x_hsi, size=(x_lidar.size(2), x_lidar.size(3)), mode='bilinear', align_corners=True)

        B, C, H, W = x_hsi.shape
        x_hsi = x_hsi.view(B, C, -1).permute(0, 2, 1)
        x_lidar = x_lidar.view(B, C, -1).permute(0, 2, 1)

        x_hsi = self.cross_modal_attn(x_hsi, x_lidar)
        x_hsi = x_hsi.permute(0, 2, 1).view(B, C, H, W)
        x_lidar = x_lidar.permute(0, 2, 1).view(B, C, H, W)
        x_fusion = self.cross_fusion(x_hsi, x_lidar)

        x_out = F.adaptive_avg_pool2d(x_fusion, 1).flatten(1)
        x_class = self.classifier(x_out)

        return x_class