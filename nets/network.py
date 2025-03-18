import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nets.unireplknet import UniRepLKNetBlock, LayerNorm, DilatedReparamBlock
import math
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            # torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            # torch.nn.init.constant_(m.bias.data, 0.0)
            pass    #   changed

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#      Kernel Prediction Network (KPN)
# ----------------------------------------
class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                # nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # # nn.BatchNorm2d(out_ch),
                # nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm
    
class LarBasic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(LarBasic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
                UniRepLKNetBlock(in_ch, 13),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                UniRepLKNetBlock(in_ch, 13),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm

class KPN(nn.Module):
    def __init__(self, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)
        
        self.conv_final = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # print(conv5.shape)
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        #print(conv7.size())
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))
        
        pred1 = self.kernel_pred(data, core, white_level, rate=1)
        pred2 = self.kernel_pred(data, core, white_level, rate=2)
        pred3 = self.kernel_pred(data, core, white_level, rate=3)
        pred4 = self.kernel_pred(data, core, white_level, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)
        
        pred = self.conv_final(pred_cat)
        
        #pred = self.kernel_pred(data, core, white_level, rate=1)
        
        return pred
    
class Attention_dec2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.task_query = nn.Parameter(torch.randn(1,48,dim))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, task_query):
        
        B, N, C = x.shape
        task_q = task_query
        
        # This is because we fix the task parameters to be of a certain dimension, so with varying batch size, we just stack up the same queries to operate on the entire batch
        # if B>1:
            
            # task_q = task_q.unsqueeze(0).repeat(B,1,1,1)
            # task_q = task_q.squeeze(1)
        # print(task_q.shape)

        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q,size= (v.shape[2],v.shape[3]))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class WeatherAtt2(nn.Module):
    def __init__(self, dim, out_dim):
        super(WeatherAtt2, self).__init__()
        self.out_dim = out_dim
        self.task_query = nn.Parameter(torch.randn(1,48,dim))
        self.WeaAtt1 = Attention_dec2(dim=dim)
        self.WeaAtt2 = Attention_dec2(dim=dim)
        self.linear1 = nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2)
        )
        self.proj1 = nn.Linear(dim, out_dim)
        self.proj2 = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.LeakyReLU(0.2)
        )
        self.act = nn.LeakyReLU(0.2)
        self.act2 = nn.Sigmoid()
        self.MaxPool = nn.AdaptiveMaxPool1d(512)
        self.AvgPool = nn.AdaptiveAvgPool1d(512)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)    #   B*C*HW
        x_pool1 = self.MaxPool(x)
        x_pool2 = self.AvgPool(x)
        x_POOL = torch.cat((x_pool1, x_pool2), dim=-1) #   B*C*1024
        x = self.linear1(x_POOL).permute(0, 2, 1).contiguous()
        x = self.WeaAtt1(x, H, W, self.task_query) #    B*HW*C
        x = self.WeaAtt2(x, H, W, self.task_query).permute(0, 2, 1).contiguous() #    B*c*512
        x  = self.proj2(x)  #   B*c*1
        x = x.permute(0, 2, 1).contiguous() #B*1*C
        x = self.proj1(x)
        x = self.act2(x)
        x = x.view(B, self.out_dim, 1, 1)

        return x

class Attention_dec(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.task_query = nn.Parameter(torch.randn(1,48,dim))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        task_q = self.task_query
        
        # This is because we fix the task parameters to be of a certain dimension, so with varying batch size, we just stack up the same queries to operate on the entire batch
        if B>1:
            
            task_q = task_q.unsqueeze(0).repeat(B,1,1,1)
            task_q = task_q.squeeze(1)

        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q,size= (v.shape[2],v.shape[3]))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class WeatherAtt(nn.Module):
    def __init__(self, dim=16):
        super(WeatherAtt, self).__init__()
        self.WeaAtt1 = Attention_dec(dim=dim)
        self.WeaAtt2 = Attention_dec(dim=dim)
        self.linear1 = nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2)
        )
        self.proj1 = nn.Linear(dim, 6)
        self.proj2 = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.LeakyReLU(0.2)
        )
        self.act = nn.LeakyReLU(0.2)
        self.act2 = nn.Sigmoid()
        self.MaxPool = nn.AdaptiveMaxPool1d(512)
        self.AvgPool = nn.AdaptiveAvgPool1d(512)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)    #   B*C*HW
        x_pool1 = self.MaxPool(x)
        x_pool2 = self.AvgPool(x)
        x_POOL = torch.cat((x_pool1, x_pool2), dim=-1) #   B*C*1024
        x = self.linear1(x_POOL).permute(0, 2, 1).contiguous()
        x = self.WeaAtt1(x, H, W) #    B*HW*C
        x = self.WeaAtt2(x, H, W).permute(0, 2, 1).contiguous() #    B*c*512
        x  = self.proj2(x)  #   B*c*1
        x = x.permute(0, 2, 1).contiguous() #B*1*C
        x = self.proj1(x)
        x = self.act2(x)
        x = x.view(B, 6, 1, 1)

        return x

class LarKPN(nn.Module):
    def __init__(self, bs, color=True, burst_length=1, blind_est=True, kernel_size=[3], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(LarKPN, self).__init__()
        self.scale = 0.5    # 控制通道数
        self.batchsize = bs
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length


        #  天气权重参数预测
        # self.linear1 = nn.Sequential(
        #     # nn.MaxPool1d(kernel_size=4, stride=4),
        #     nn.AdaptiveMaxPool1d(256),
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(0.2),
        #     # nn.InstanceNorm1d(self.batchsize),
        #     nn.Linear(64,1),
        #     nn.Sigmoid()
        # )
        
        self.WeaAttn = WeatherAtt(dim=16)
        self.fuse = nn.Conv2d(6, 3, 3, 1, 1)


        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        # self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        # self.conv2 = LarBasic(64, 128, channel_att=False, spatial_att=False)
        # self.conv3 = LarBasic(128, 256, channel_att=False, spatial_att=False)
        # self.conv4 = LarBasic(256, 512, channel_att=False, spatial_att=False)
        # self.conv5 = LarBasic(512, 512, channel_att=False, spatial_att=False)
        # # 6~8层要先上采样再卷积
        # self.conv6a = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv7a = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv8a = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        # self.outca = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        # self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)
        
        # self.conv_finala = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

        # self.conv6b = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv7b = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv8b = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        # self.outcb = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        
        # self.conv_finalb = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

        # 减少模型通道数

        self.conv1 = Basic(in_channel, 32, channel_att=False, spatial_att=False)
        self.conv2 = LarBasic(32, 64, channel_att=False, spatial_att=False)
        self.conv3 = LarBasic(64, 128, channel_att=False, spatial_att=False)
        self.conv4 = LarBasic(128, 256, channel_att=False, spatial_att=False)
        self.conv5 = LarBasic(256, 256, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6a = Basic(256+256, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7a = Basic(128+256, 128, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8a = Basic(128+64, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outca = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)
        
        self.conv_finala = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.conv6b = Basic(256+256, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7b = Basic(128+256, 128, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8b = Basic(128+64, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outcb = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        
        self.conv_finalb = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        # print(data_with_est.shape)
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))


        # pr = conv4[:, 0, :, :]
        # pr = pr.reshape(conv4.shape[0], -1)
        # # print(pr.shape)
        # pr = self.linear1(pr)
        # pr = pr.view(-1, 1, 1, 1)

        pr = conv5[:, :16, :, :]
        pr = self.WeaAttn(pr)
        # print(pr)
        # 开始上采样  同时要进行skip connection
        conv6a = self.conv6a(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7a = self.conv7a(torch.cat([conv3, F.interpolate(conv6a, scale_factor=2, mode=self.upMode)], dim=1))
        #print(conv7.size())
        conv8a = self.conv8a(torch.cat([conv2, F.interpolate(conv7a, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        corea = self.outca(F.interpolate(conv8a, scale_factor=2, mode=self.upMode))

        # corea = self.conv6a(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        # corea = self.conv7a(torch.cat([conv3, F.interpolate(corea, scale_factor=2, mode=self.upMode)], dim=1))
        # #print(conv7.size())
        # corea = self.conv8a(torch.cat([conv2, F.interpolate(corea, scale_factor=2, mode=self.upMode)], dim=1))
        # # return channel K*K*N
        # corea = self.outca(F.interpolate(corea, scale_factor=2, mode=self.upMode))

        
        pred1a = self.kernel_pred(data, corea, white_level, rate=1)
        pred2a = self.kernel_pred(data, corea, white_level, rate=2)
        pred3a = self.kernel_pred(data, corea, white_level, rate=3)
        pred4a = self.kernel_pred(data, corea, white_level, rate=4)

        pred_cata = torch.cat([torch.cat([torch.cat([pred1a, pred2a], dim=1), pred3a], dim=1), pred4a], dim=1)
        # preda = torch.cat(
        #     [self.kernel_pred(data, corea, white_level, rate=i) for i in range(1, 5)], 
        #     dim=1
        #     )
        
        preda = self.conv_finala(pred_cata)

        conv6b = self.conv6b(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7b = self.conv7b(torch.cat([conv3, F.interpolate(conv6b, scale_factor=2, mode=self.upMode)], dim=1))
        #print(conv7.size())
        conv8b = self.conv8b(torch.cat([conv2, F.interpolate(conv7b, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        coreb = self.outcb(F.interpolate(conv8b, scale_factor=2, mode=self.upMode))

        # coreb = self.conv6b(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        # coreb = self.conv7b(torch.cat([conv3, F.interpolate(coreb, scale_factor=2, mode=self.upMode)], dim=1))
        # #print(conv7.size())
        # coreb = self.conv8b(torch.cat([conv2, F.interpolate(coreb, scale_factor=2, mode=self.upMode)], dim=1))
        # # return channel K*K*N
        # coreb = self.outcb(F.interpolate(coreb, scale_factor=2, mode=self.upMode))

        
        pred1b = self.kernel_pred(data, coreb, white_level, rate=1)
        pred2b = self.kernel_pred(data, coreb, white_level, rate=2)
        pred3b = self.kernel_pred(data, coreb, white_level, rate=3)
        pred4b = self.kernel_pred(data, coreb, white_level, rate=4)

        pred_catb = torch.cat([torch.cat([torch.cat([pred1b, pred2b], dim=1), pred3b], dim=1), pred4b], dim=1)
        # predb = torch.cat(
        #     [self.kernel_pred(data, coreb, white_level, rate=i) for i in range(1, 5)], 
        #     dim=1
        #     )
        
        predb = self.conv_finalb(pred_catb)

        predab = torch.cat((preda,predb), dim=1)
        # predab_att = predab * pr
        pred = self.fuse(predab * pr)

        # pred = pr * preda + (1 - pr) * predb

        return pred


    

class LarKPN2(nn.Module):
    def __init__(self, bs, color=True, burst_length=1, blind_est=True, kernel_size=[3], sep_conv=False,
                 channel_att=True, spatial_att=True, upMode='bilinear', core_bias=False):
        super(LarKPN2, self).__init__()
        self.kernel_num = 4   
        self.scale = 0.5    # 控制通道数
        self.batchsize = bs
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length * self.kernel_num
        if core_bias:
            out_channel += (3 if color else 1) * burst_length


        #  天气权重参数预测
        # self.linear1 = nn.Sequential(
        #     # nn.MaxPool1d(kernel_size=4, stride=4),
        #     nn.AdaptiveMaxPool1d(256),
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(0.2),
        #     # nn.InstanceNorm1d(self.batchsize),
        #     nn.Linear(64,1),
        #     nn.Sigmoid()
        # )
        
        self.WeaAttn = WeatherAtt2(16, 12*self.kernel_num)
        self.fuse = nn.Conv2d(6, 3, 3, 1, 1)


        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        # self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        # self.conv2 = LarBasic(64, 128, channel_att=False, spatial_att=False)
        # self.conv3 = LarBasic(128, 256, channel_att=False, spatial_att=False)
        # self.conv4 = LarBasic(256, 512, channel_att=False, spatial_att=False)
        # self.conv5 = LarBasic(512, 512, channel_att=False, spatial_att=False)
        # # 6~8层要先上采样再卷积
        # self.conv6a = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv7a = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv8a = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        # self.outca = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        # self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)
        
        # self.conv_finala = nn.Conv2d(in_channels=12*self.kernel_num, out_channels=3, kernel_size=3, stride=1, padding=1)

        # self.conv6b = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv7b = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv8b = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        # self.outcb = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        
        # self.conv_finalb = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

        # 减少模型通道数

        self.conv1 = Basic(in_channel, 32, channel_att=False, spatial_att=False)
        self.conv2 = LarBasic(32, 64, channel_att=False, spatial_att=False)
        self.conv3 = LarBasic(64, 128, channel_att=False, spatial_att=False)
        self.conv4 = LarBasic(128, 256, channel_att=False, spatial_att=False)
        self.conv5 = LarBasic(256, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6a = Basic(256+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7a = Basic(128+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8a = Basic(64+256, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outca = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)
        
        self.conv_finala = nn.Conv2d(in_channels=12*self.kernel_num, out_channels=3, kernel_size=3, stride=1, padding=1)

        # self.conv6b = Basic(256+256, 256, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv7b = Basic(128+256, 128, channel_att=channel_att, spatial_att=spatial_att)
        # self.conv8b = Basic(128+64, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        # self.outcb = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        
        # self.conv_finalb = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        # print(data_with_est.shape)
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))


        # pr = conv4[:, 0, :, :]
        # pr = pr.reshape(conv4.shape[0], -1)
        # # print(pr.shape)
        # pr = self.linear1(pr)
        # pr = pr.view(-1, 1, 1, 1)

        pr = conv5[:, :16, :, :]
        pr = self.WeaAttn(pr, )
        # print(pr)
        # 开始上采样  同时要进行skip connection
        conv6a = self.conv6a(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7a = self.conv7a(torch.cat([conv3, F.interpolate(conv6a, scale_factor=2, mode=self.upMode)], dim=1))
        #print(conv7.size())
        conv8a = self.conv8a(torch.cat([conv2, F.interpolate(conv7a, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        corea = self.outca(F.interpolate(conv8a, scale_factor=2, mode=self.upMode))

        # corea = self.conv6a(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        # corea = self.conv7a(torch.cat([conv3, F.interpolate(corea, scale_factor=2, mode=self.upMode)], dim=1))
        # #print(conv7.size())
        # corea = self.conv8a(torch.cat([conv2, F.interpolate(corea, scale_factor=2, mode=self.upMode)], dim=1))
        # # return channel K*K*N
        # corea = self.outca(F.interpolate(corea, scale_factor=2, mode=self.upMode))

        
        # pred1a = self.kernel_pred(data, corea, white_level, rate=1)
        # pred2a = self.kernel_pred(data, corea, white_level, rate=2)
        # pred3a = self.kernel_pred(data, corea, white_level, rate=3)
        # pred4a = self.kernel_pred(data, corea, white_level, rate=4)

        # pred_cata = torch.cat([torch.cat([torch.cat([pred1a, pred2a], dim=1), pred3a], dim=1), pred4a], dim=1)
        pred_cata = torch.cat(
            [self.kernel_pred(data, corea[:,j*3:(j+1)*3,:,:], white_level, rate=i) for i in range(1, 5) for j in range(self.kernel_num)], 
            dim=1
            )
        pred_cata = pr * pred_cata
        pred = self.conv_finala(pred_cata)

        # conv6b = self.conv6b(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        # conv7b = self.conv7b(torch.cat([conv3, F.interpolate(conv6b, scale_factor=2, mode=self.upMode)], dim=1))
        # #print(conv7.size())
        # conv8b = self.conv8b(torch.cat([conv2, F.interpolate(conv7b, scale_factor=2, mode=self.upMode)], dim=1))
        # # return channel K*K*N
        # coreb = self.outcb(F.interpolate(conv8b, scale_factor=2, mode=self.upMode))

        # # coreb = self.conv6b(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        # # coreb = self.conv7b(torch.cat([conv3, F.interpolate(coreb, scale_factor=2, mode=self.upMode)], dim=1))
        # # #print(conv7.size())
        # # coreb = self.conv8b(torch.cat([conv2, F.interpolate(coreb, scale_factor=2, mode=self.upMode)], dim=1))
        # # # return channel K*K*N
        # # coreb = self.outcb(F.interpolate(coreb, scale_factor=2, mode=self.upMode))

        
        # pred1b = self.kernel_pred(data, coreb, white_level, rate=1)
        # pred2b = self.kernel_pred(data, coreb, white_level, rate=2)
        # pred3b = self.kernel_pred(data, coreb, white_level, rate=3)
        # pred4b = self.kernel_pred(data, coreb, white_level, rate=4)

        # pred_catb = torch.cat([torch.cat([torch.cat([pred1b, pred2b], dim=1), pred3b], dim=1), pred4b], dim=1)
        # # predb = torch.cat(
        # #     [self.kernel_pred(data, coreb, white_level, rate=i) for i in range(1, 5)], 
        # #     dim=1
        # #     )
        
        # predb = self.conv_finalb(pred_catb)

        # predab = torch.cat((preda,predb), dim=1)
        # # predab_att = predab * pr
        # pred = self.fuse(predab * pr)

        # pred = pr * preda + (1 - pr) * predb

        return pred

class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0, rate=1):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                padding_num = (K//2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                for i in range(0, K):
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            # print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        # print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        #print("pred_img_i", pred_img_i.size())
        # N = 1
        pred_img_i = pred_img_i.squeeze(2)
        #print("pred_img_i", pred_img_i.size())
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
        # print('white_level', white_level.size())
        pred_img_i = pred_img_i / white_level
        #pred_img = torch.mean(pred_img_i, dim=1, keepdim=True)
        # print('pred_img:', pred_img.size())
        # print('pred_img_i:', pred_img_i.size())
        return pred_img_i

class LossFunc(nn.Module):
    """
    loss function of KPN
    """
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)

class LossBasic(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth) + \
               self.l1_loss(self.gradient(pred), self.gradient(ground_truth))

class LossAnneal(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss

class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class WeatherAtt3(nn.Module):
    def __init__(self, dim):
        super(WeatherAtt3, self).__init__()
        self.task_query = nn.Parameter(torch.randn(1,48,dim))  #   B*N*C
        self.WeaAtt1 = Attention_dec2(dim=dim)
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            # nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=2),
            nn.Flatten(start_dim=1),
            nn.Linear(dim, dim),
            nn.Softmax(dim=1),
        )
        # self.act = nn.LeakyReLU(0.2)
        # self.act2 = nn.Sigmoid()
        # self.MaxPool = nn.AdaptiveMaxPool1d(128)
        # self.AvgPool = nn.AdaptiveAvgPool1d(128)
        self.AvgPool2 = nn.AdaptiveAvgPool1d(1)
        

    def forward(self, x):
        o_x = x
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)    #   B*C*HW
        # print((self.seq(x).reshape(B, 1, C)*self.task_query).shape)
        x = self.WeaAtt1(x.permute(0,2,1), H, W, self.seq(x).reshape(B, 1, C) * self.task_query) #    B*HW*C
        # x = self.WeaAtt1(x.permute(0,2,1), H, W, self.task_query) #    B*HW*C
        x  = x.permute(0, 2, 1) #   B*C*N
        x = self.AvgPool2(x).reshape(B,C,1,1)    #   B*C*1
        # x = self.act2(x)


        return x * o_x  

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

     

class EDLKBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
        #                        bias=True)
        self.conv2 = DilatedReparamBlock(dw_channel, 13, deploy=False,
                                              use_sync_bn=False,
                                              attempt_use_lk_impl=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma



class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=2, enc_blk_nums=[2, 2, 2, 2], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[EDLKBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        # self.middle_blks = \
        #     nn.Sequential(
        #         *[NAFBlock(chan) for _ in range((middle_blk_num)//2)], 
        #         WeatherAtt3(chan),
        #         *[NAFBlock(chan) for _ in range((middle_blk_num)//2)]
        #     )

        for idx, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            # self.decoders.append(
            #     nn.Sequential(
            #         # WeatherAtt3(chan),
            #         *[NAFBlock(chan) for _ in range(num)]
            #     )
            # )
            # print(idx)
            if idx < 2:
                # 前三个解码器块前添加 WeatherAtt3
                decoder = nn.Sequential(
                    WeatherAtt3(chan),
                    *[NAFBlock(chan) for _ in range(num)]
                )
            else:
                # 其余解码器块仅添加 NAFBlock
                decoder = nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            
            self.decoders.append(decoder)

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
