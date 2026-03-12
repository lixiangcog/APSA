import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

def configure_model_with_bn(model):
    """
    配置模型用于Tent-EMA TTA（包含BN适应）
    
    核心改动：
    1. 冻结卷积层权重
    2. 解冻BN的affine参数（gamma, beta）
    3. 设置BN为训练模式（更新running stats）
    """
    # 1. 冻结非BN参数
    for name, param in model.named_parameters():
        if 'bn' not in name and 'vida' not in name and 'data_adaptor' not in name and 'data_prompt' not in name:
            param.requires_grad = False
    
    # 2. 解冻BN的affine参数
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            # 设置为训练模式（会更新running_mean和running_var）
            module.train()
            # 确保affine参数可训练
            if module.affine:
                module.weight.requires_grad = True
                module.bias.requires_grad = True
            # 启用统计量追踪
            module.track_running_stats = True
            # 使用指数移动平均更新统计量（momentum=0.1是PyTorch默认值）
            module.momentum = 0.1
    
    # 3. 确保ViDA参数可训练
    for name, param in model.named_parameters():
        if 'vida' in name or 'data_adaptor' in name or 'data_prompt' in name:
            param.requires_grad = True
    
    return model


def collect_trainable_params(model):
    """收集所有可训练参数"""
    params = []
    names = []

    # ViDA参数
    for name, param in model.named_parameters():
        if 'vida' in name and param.requires_grad:
            params.append(param)
            names.append(name)

    # BN参数
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            if module.affine:
                params.append(module.weight)
                params.append(module.bias)
                names.append(f'bn.weight')
                names.append(f'bn.bias')

    return params, names


def setup_optimizer(model, lr=1e-4):
    """
    设置优化器（分组学习率）
    
    策略：
    - ViDA参数: lr = 1e-4
    - BN参数: lr = 1e-3 (更高学习率以快速适应)
    """
    vida_params = []
    bn_params = []
    
    # 收集ViDA参数
    for name, param in model.named_parameters():
        if 'vida' in name and param.requires_grad:
            vida_params.append(param)
    
    # 收集BN参数
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            if module.affine:
                bn_params.extend([module.weight, module.bias])
    
    # 分组优化器
    optimizer = optim.Adam([
        {'params': vida_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': bn_params, 'lr': lr * 10, 'weight_decay': 0.0}  # BN不用weight decay
    ])
    
    return optimizer
