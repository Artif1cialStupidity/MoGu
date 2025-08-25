# defenses/victim/utils/my_utils.py (REVISED)

import sys
import os
import torch.nn as nn
sys.path.append(os.getcwd())
from defenses import datasets
from defenses.models import zoo

class UnifiedBackbone(nn.Module):
    """一个包装器，统一VGG和ResNet的接口"""
    def __init__(self, original_model):
        super().__init__()
        if 'vgg' in original_model.__class__.__name__.lower():
            self.features = nn.Sequential(
                original_model._features,
                nn.Flatten(1)
            )
            self.last_linear = original_model.classifier
        elif 'resnet' in original_model.__class__.__name__.lower():
            self.features = nn.Sequential(
                original_model.features,
                original_model.avgpool
            )
            self.last_linear = original_model.last_linear
        else:
            # 默认行为，适用于已经有 .features 和 .last_linear 的模型
            self.features = original_model.features
            self.last_linear = original_model.last_linear

def get_classifier(args, model_arch_name, pretrained=False, num_classes=10):
    pretrain_flag = 'imagenet' if pretrained else None
    dataset_name = getattr(args, 'dataset', 'CIFAR10') # 使用大写
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    
    print(f"[get_classifier] Loading model '{model_arch_name}' for modelfamily '{modelfamily}' with {num_classes} classes.")
    
    original_model = zoo.get_net(model_arch_name, modelfamily, pretrained=pretrain_flag, num_classes=num_classes)
    
    # **修正点**: 我们不再需要对返回的模型做任何操作，因为DynamicAttentionModel内部会处理
    return original_model
