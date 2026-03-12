import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vida_adapter import ViDAConvAdapter
from models.unet import UNet

# ============ SaveFeatures 用于记录BN统计量 ============
class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


class CrossAttention(nn.Module):
    """简化的交叉注意力模块"""
    def __init__(self, C, K, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (C // num_heads) ** -0.5
        
        self.q_proj = nn.Conv2d(C, C, 1)
        self.k_proj = nn.Conv2d(K, C, 1)
        self.v_proj = nn.Conv2d(K, C, 1)
        
    def forward(self, x_q, x_kv):
        batch_size, channels, height, width = x_q.shape
        q = self.q_proj(x_q).view(batch_size, self.num_heads, channels // self.num_heads, -1)
        k = self.k_proj(x_kv).view(batch_size, self.num_heads, channels // self.num_heads, -1)
        v = self.v_proj(x_kv).view(batch_size, self.num_heads, channels // self.num_heads, -1)
        
        attn_weights = torch.softmax(torch.matmul(q.transpose(-2, -1), k) * self.scale, dim=-1)
        
        # Top-k稀疏注意力
        topk = max(1, int(0.1 * attn_weights.size(-1)))
        _, sorted_indices = torch.topk(attn_weights, topk, dim=-1)
        mask = torch.zeros_like(attn_weights)
        mask.scatter_(-1, sorted_indices, 1)
        attn_weights = attn_weights * mask
        
        attn_output = torch.matmul(attn_weights, v.transpose(-2, -1))
        attn_output = attn_output.transpose(-2, -1).contiguous().view(batch_size, channels, height, width)
        
        return x_q + attn_output, attn_weights


class UNet_SPTTA(nn.Module):
    def __init__(self, pretrained_path, patch_size=(512, 512), resnet='resnet34', 
                 num_classes=2, pretrained=False):
        super().__init__()
        
        # ============ Data Prompt & Adaptor ============
        data_prompt = torch.zeros(1, 512, patch_size[0]//32, patch_size[1]//32)
        self.data_prompt = nn.Parameter(data_prompt)
        self.prompt2feature = CrossAttention(C=512, K=512)
        
        self.data_adaptor = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1)
        )
        
        # ============ UNet Backbone ============
        self.unet = UNet(resnet=resnet, num_classes=num_classes, pretrained=pretrained)
        pretrained_params = torch.load(pretrained_path)
        self.unet.load_state_dict(pretrained_params['model_state_dict'])
        
        # ============ ViDA Adapters ============
        self.vida_head = ViDAConvAdapter(channels=512, high_dim=64, low_dim=8)
        
        # ============ 关键修改：解冻BN层 ============
        # 1. 冻结卷积层权重
        for name, param in self.unet.named_parameters():
            if 'bn' not in name and 'downsample.1' not in name:  # 保留BN
                param.requires_grad = False
        
        # 2. 明确设置BN层为训练模式（CRITICAL!）
        for module in self.unet.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.train()  # 强制训练模式
                module.requires_grad_(True)  # 确保可训练
        
        # ============ 记录BN特征（用于监控） ============
        self.bn_f = []
        # ResNet的所有BN层
        if hasattr(self.unet, 'rn'):
            # Initial conv的BN
            if len(self.unet.rn) > 0:
                self.bn_f.append(SaveFeatures(self.unet.rn[0]))
            
            # ResNet blocks
            for layer_idx in [4, 5, 6, 7]:
                if layer_idx < len(self.unet.rn):
                    layer = self.unet.rn[layer_idx]
                    for block in layer:
                        # conv1, conv2的BN
                        if hasattr(block, 'conv1'):
                            self.bn_f.append(SaveFeatures(block.conv1))
                        if hasattr(block, 'conv2'):
                            self.bn_f.append(SaveFeatures(block.conv2))
                        # downsample的BN
                        if hasattr(block, 'downsample') and block.downsample is not None:
                            if len(block.downsample) > 0:
                                self.bn_f.append(SaveFeatures(block.downsample[0]))
        
        # UNet decoder的BN层
        for up_name in ['up1', 'up2', 'up3', 'up4']:
            if hasattr(self.unet, up_name):
                up_block = getattr(self.unet, up_name)
                if hasattr(up_block, 'tr_conv'):
                    self.bn_f.append(SaveFeatures(up_block.tr_conv))
                if hasattr(up_block, 'x_conv'):
                    self.bn_f.append(SaveFeatures(up_block.x_conv))

    def forward(self, x, training=False):
        # Data perturbation
        perturbation = self.data_adaptor(x)
        adapted_x = x + perturbation
        
        # Encoder (ResNet with ADAPTIVE BN)
        bo_fea = self.unet.encoder(adapted_x)
        
        # Prompt matching
        data_prompt = self.data_prompt.repeat(x.shape[0], 1, 1, 1)
        adapted_bo_fea, attn_weights = self.prompt2feature(x_q=bo_fea, x_kv=data_prompt)
        
        # ViDA adapter at bottleneck
        adapted_bo_fea = self.vida_head(adapted_bo_fea)
        
        # Decoder (with adaptive BN)
        x = self.unet.up1(adapted_bo_fea, self.unet.sfs[3].features)
        x = self.unet.up2(x, self.unet.sfs[2].features)
        x = self.unet.up3(x, self.unet.sfs[1].features)
        x = self.unet.up4(x, self.unet.sfs[0].features)
        output = self.unet.up5(x)
        
        if training:
            return output, self.bn_f, perturbation, attn_weights
        else:
            return output

    def get_bn_params(self):
        """获取所有BN层的参数（用于优化器）"""
        bn_params = []
        for module in self.unet.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                bn_params.extend(list(module.parameters()))
        return bn_params
