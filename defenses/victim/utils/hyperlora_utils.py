# defenses/victim/utils/hyperlora_utils.py (完整最终版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .my_utils import get_classifier
import os

class DynamicAttentionModel(nn.Module):
    def __init__(self, args, num_classes=10):
        super(DynamicAttentionModel, self).__init__()
        self.args = args
        self.num_basis_weights = getattr(args, 'num_basis_weights', 8)
        self.z_dim = getattr(args, 'z_dim', 64)
        self.seed_pool_size = getattr(args, 'seed_pool_size', 2048)
        
        # --- 1. Backbone Loading ---
        print(f"[HyperLoRA] Initializing backbone for {args.dataset}...")
        backbone = get_classifier(args, args.model, pretrained=False, num_classes=num_classes)
        backbone_ckpt_path = os.path.abspath(args.backbone_ckpt)
        print(f"[HyperLoRA] Loading backbone weights from: {backbone_ckpt_path}")
        
        checkpoint = torch.load(backbone_ckpt_path, map_location='cpu', weights_only=False)
        
        # 健壮地加载状态字典
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict_to_load = checkpoint['state_dict']
        else:
            state_dict_to_load = checkpoint
        backbone.load_state_dict(state_dict_to_load)

        # --- 2. Component Definition and Freezing ---
        model_name_lower = args.model.lower()
        if 'vgg' in model_name_lower:
            self.feature_dim = backbone.classifier.in_features
            print(f"[HyperLoRA] Feature dimension from VGG backbone: {self.feature_dim}")
            self.feature_extractor = nn.Sequential(backbone._features, nn.Flatten(1))
            self.original_classifier = backbone.classifier
        elif 'resnet' in model_name_lower:
            self.feature_dim = backbone.fc.in_features
            print(f"[HyperLoRA] Feature dimension from ResNet backbone: {self.feature_dim}")
            feature_modules = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*feature_modules, nn.Flatten(1))
            self.original_classifier = backbone.fc
        else:
            raise NotImplementedError(f"Backbone '{args.model}' structure adapter not implemented.")

        for param in self.feature_extractor.parameters(): param.requires_grad = False
        for param in self.original_classifier.parameters(): param.requires_grad = False
        print("[HyperLoRA] Backbone feature extractor and original classifier frozen.")

        # --- 3. Trainable Modules (静态Acc + 动态Div) ---
        # 静态准确性路径：一组可直接训练的系数
        self.static_accuracy_coeffs = nn.Parameter(torch.randn(1, self.num_basis_weights))
        
        # 动态多样性路径：一个超网络
        self.hypernetwork_diversity = nn.Sequential(
            nn.Linear(self.z_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, self.num_basis_weights), 
            nn.Softmax(dim=1)
        )
        
        # 共享的基底权重
        self.basis_weights_q = nn.ParameterList([nn.Parameter(torch.randn(self.feature_dim, self.feature_dim)) for _ in range(self.num_basis_weights)])
        self.basis_weights_k = nn.ParameterList([nn.Parameter(torch.randn(self.feature_dim, self.feature_dim)) for _ in range(self.num_basis_weights)])
        self.basis_weights_v = nn.ParameterList([nn.Parameter(torch.randn(self.feature_dim, self.feature_dim)) for _ in range(self.num_basis_weights)])
        
        self.seed_embedding = nn.Embedding(self.seed_pool_size, self.z_dim)
        
        # 新的可训练分类器
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        print("[HyperLoRA] STATIC Accuracy Path & DYNAMIC Diversity Path Initialized.")

        # --- 4. Parameter Grouping ---
        self.static_accuracy_coeffs.param_group = 'accuracy'
        for p in self.classifier.parameters(): p.param_group = 'accuracy'
        
        for p in self.hypernetwork_diversity.parameters(): p.param_group = 'diversity'
        for p in self.seed_embedding.parameters(): p.param_group = 'diversity'
        
        for p_list in [self.basis_weights_q, self.basis_weights_k, self.basis_weights_v]:
            for p in p_list: p.param_group = 'basis'

    def load_defense_modules(self, state_dict):
        # 加载所有可训练参数
        defense_state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict()}
        self.load_state_dict(defense_state_dict, strict=False)
        print("[HyperLoRA] Defense-related modules loaded successfully.")

    def _synthesize_attention_weights(self, z_embed):
        # 静态路径系数
        coeffs_acc = F.softmax(self.static_accuracy_coeffs, dim=1)
        batch_size = z_embed.shape[0]
        coeffs_acc_expanded = coeffs_acc.expand(batch_size, -1)
        
        # 动态路径系数
        coeffs_div = self.hypernetwork_diversity(z_embed)
        
        # 融合系数
        final_coeffs = coeffs_acc_expanded + coeffs_div
        
        w_q_stack = torch.stack(list(self.basis_weights_q), dim=0)
        w_k_stack = torch.stack(list(self.basis_weights_k), dim=0)
        w_v_stack = torch.stack(list(self.basis_weights_v), dim=0)

        final_w_q = torch.einsum('bi, ijk -> bjk', final_coeffs, w_q_stack)
        final_w_k = torch.einsum('bi, ijk -> bjk', final_coeffs, w_k_stack)
        final_w_v = torch.einsum('bi, ijk -> bjk', final_coeffs, w_v_stack)

        return final_w_q, final_w_k, final_w_v

    def forward_acc_only(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        
        features_seq = features.unsqueeze(1)
        
        coeffs_acc = F.softmax(self.static_accuracy_coeffs, dim=1)
        batch_size = x.shape[0]
        coeffs_acc_expanded = coeffs_acc.expand(batch_size, -1)
        
        w_q_stack = torch.stack(list(self.basis_weights_q), dim=0)
        w_k_stack = torch.stack(list(self.basis_weights_k), dim=0)
        w_v_stack = torch.stack(list(self.basis_weights_v), dim=0)
        
        final_w_q = torch.einsum('bi, ijk -> bjk', coeffs_acc_expanded, w_q_stack)
        final_w_k = torch.einsum('bi, ijk -> bjk', coeffs_acc_expanded, w_k_stack)
        final_w_v = torch.einsum('bi, ijk -> bjk', coeffs_acc_expanded, w_v_stack)

        q = torch.bmm(features_seq, final_w_q)
        k = torch.bmm(features_seq, final_w_k)
        v = torch.bmm(features_seq, final_w_v)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.feature_dim ** 0.5)
        attn_map = F.softmax(attn_scores, dim=-1)
        attended_features = torch.bmm(attn_map, v)

        pooled_features = attended_features.squeeze(1)
        logits = self.classifier(pooled_features)
        
        return logits, pooled_features

    def forward(self, x, seeds):
        with torch.no_grad():
            features = self.feature_extractor(x)
        
        z_embed = self.seed_embedding(seeds)
        return self.forward_from_z_embed(features, z_embed)

    def forward_from_z_embed(self, features, z_embed):
        features_seq = features.unsqueeze(1)
        
        final_w_q, final_w_k, final_w_v = self._synthesize_attention_weights(z_embed)

        q = torch.bmm(features_seq, final_w_q)
        k = torch.bmm(features_seq, final_w_k)
        v = torch.bmm(features_seq, final_w_v)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.feature_dim ** 0.5)
        attn_map = F.softmax(attn_scores, dim=-1)
        attended_features = torch.bmm(attn_map, v)

        pooled_features = attended_features.squeeze(1)
        logits = self.classifier(pooled_features)
        
        return logits, attn_map
