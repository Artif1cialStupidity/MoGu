# defenses/victim/utils/hyperlora_utils.py (REVISED)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .my_utils import get_classifier
import os
import random

# ... (之前的所有代码保持不变，直到 DynamicAttentionModel 类定义) ...

class DynamicAttentionModel(nn.Module):
    def __init__(self, args, num_classes=10):
        super(DynamicAttentionModel, self).__init__()
        self.args = args
        self.num_basis_weights = getattr(args, 'num_basis_weights', 8)
        self.lora_rank = getattr(args, 'lora_rank', 32)
        self.z_dim = getattr(args, 'z_dim', 64)
        self.seed_pool_size = getattr(args, 'seed_pool_size', 2048)
        
        # --- 1. Backbone Loading ---
        print(f"[HyperLoRA] Initializing backbone for {args.dataset}...")
        backbone = get_classifier(args, args.model, pretrained=False, num_classes=num_classes)
        backbone_ckpt_path = os.path.abspath(args.backbone_ckpt)
        print(f"[HyperLoRA] Loading backbone weights from: {backbone_ckpt_path}")
        
        checkpoint = torch.load(backbone_ckpt_path, map_location='cpu', weights_only=False)
        backbone.load_state_dict(checkpoint['state_dict'])

        # --- 2. Component Definition and Freezing ---
        if 'vgg' in args.model.lower():
            self.feature_dim = backbone.classifier.in_features
            print(f"[HyperLoRA] Feature dimension from VGG backbone: {self.feature_dim}")
            self.feature_extractor = nn.Sequential(backbone._features, nn.Flatten(1))
            self.original_classifier = backbone.classifier
        elif 'resnet' in args.model.lower():
            self.feature_dim = backbone.last_linear.in_features
            print(f"[HyperLoRA] Feature dimension from ResNet backbone: {self.feature_dim}")
            self.feature_extractor = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten(1))
            self.original_classifier = backbone.last_linear
        else:
            raise NotImplementedError(f"Backbone '{args.model}' structure adapter not implemented.")

        for param in self.feature_extractor.parameters(): param.requires_grad = False
        for param in self.original_classifier.parameters(): param.requires_grad = False
        print("[HyperLoRA] Backbone modules frozen.")

        # --- 3. Trainable Modules ---
        self.hypernetwork_accuracy = nn.Sequential(nn.Linear(self.z_dim, 256), nn.ReLU(), nn.Linear(256, self.num_basis_weights), nn.Softmax(dim=1))
        self.hypernetwork_diversity = nn.Sequential(nn.Linear(self.z_dim, 256), nn.ReLU(), nn.Linear(256, self.num_basis_weights), nn.Softmax(dim=1))
        self.basis_A = nn.ParameterList([nn.Parameter(torch.randn(self.feature_dim, self.lora_rank)) for _ in range(self.num_basis_weights)])
        self.basis_B = nn.ParameterList([nn.Parameter(torch.zeros(self.lora_rank, self.feature_dim)) for _ in range(self.num_basis_weights)])
        self.seed_embedding = nn.Embedding(self.seed_pool_size, self.z_dim)
        print("[HyperLoRA] All-dynamic trainable modules initialized.")

        # --- 4. Parameter Grouping ---
        for p in self.hypernetwork_accuracy.parameters(): p.param_group = 'accuracy'
        for p in self.hypernetwork_diversity.parameters(): p.param_group = 'diversity'
        for p in self.seed_embedding.parameters(): p.param_group = 'diversity'
        for p_list in [self.basis_A, self.basis_B]:
            for p in p_list: p.param_group = 'basis'

    # --- [新增] 方法: 只加载防御模块的权重 ---
    def load_defense_modules(self, state_dict):
        # 创建一个只包含可训练模块键的 state_dict
        defense_state_dict = {k: v for k, v in state_dict.items() if 'hypernetwork' in k or 'basis' in k or 'seed_embedding' in k}
        self.load_state_dict(defense_state_dict, strict=False)
        print("[HyperLoRA] Defense-related modules (HyperNets, Basis, Embedding) loaded successfully.")

    # ... (后续方法 forward, _synthesize_lora 等保持不变) ...
    def _synthesize_lora(self, z_embed):
        coeffs_acc = self.hypernetwork_accuracy(z_embed)
        w_A_stack = torch.stack(list(self.basis_A), dim=0)
        w_B_stack = torch.stack(list(self.basis_B), dim=0)
        lora_A_acc = torch.einsum('bi, ijk -> bjk', coeffs_acc, w_A_stack)
        lora_B_acc = torch.einsum('bi, ijk -> bjk', coeffs_acc, w_B_stack)
        coeffs_div = self.hypernetwork_diversity(z_embed)
        lora_A_div = torch.einsum('bi, ijk -> bjk', coeffs_div, w_A_stack)
        lora_B_div = torch.einsum('bi, ijk -> bjk', coeffs_div, w_B_stack)
        final_lora_A = lora_A_acc + lora_A_div
        final_lora_B = lora_B_acc + lora_B_div
        return final_lora_A, final_lora_B

    def forward_acc_only(self, x, seeds):
        with torch.no_grad():
            pooled_features = self.feature_extractor(x)
        z_embed = self.seed_embedding(seeds)
        coeffs_acc = self.hypernetwork_accuracy(z_embed)
        w_A_stack = torch.stack(list(self.basis_A), dim=0)
        w_B_stack = torch.stack(list(self.basis_B), dim=0)
        lora_A_acc = torch.einsum('bi, ijk -> bjk', coeffs_acc, w_A_stack)
        lora_B_acc = torch.einsum('bi, ijk -> bjk', coeffs_acc, w_B_stack)
        lora_half = torch.bmm(pooled_features.unsqueeze(1), lora_A_acc)
        lora_adjustment = torch.bmm(lora_half, lora_B_acc).squeeze(1)
        final_features = pooled_features + lora_adjustment
        logits = self.original_classifier(final_features)
        return logits, final_features

    def forward(self, x, seeds):
        with torch.no_grad():
            pooled_features = self.feature_extractor(x)
        z_embed = self.seed_embedding(seeds)
        return self.forward_from_z_embed(pooled_features, z_embed)

    def forward_from_z_embed(self, pooled_features, z_embed):
        final_lora_A, final_lora_B = self._synthesize_lora(z_embed)
        lora_half = torch.bmm(pooled_features.unsqueeze(1), final_lora_A)
        lora_adjustment = torch.bmm(lora_half, final_lora_B).squeeze(1)
        final_features = pooled_features + lora_adjustment
        logits = self.original_classifier(final_features)
        return logits, final_features
