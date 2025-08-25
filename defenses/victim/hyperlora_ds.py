# defenses/victim/hyperlora_ds.py (REVISED)

import torch
import torch.nn.functional as F
import os
import os.path as osp
import json
import pickle
import random

from .blackbox import Blackbox
from .utils.hyperlora_utils import DynamicAttentionModel

class HyperLoRA_DS(Blackbox):
    def __init__(self, model, hyperlora_ckpt_path, seed_pool_size, samples_per_seed, **kwargs):
        super().__init__(model=model, **kwargs)
        print(f"=> Initializing HyperLoRA_DS Defense...")
        self.require_xinfo = False

        class DefenseArgs:
            def __init__(self, ckpt, seed_pool_size, dataset, model, backbone_ckpt):
                self.ckpt = ckpt
                self.seed_pool_size = seed_pool_size
                self.dataset = dataset
                self.model = model
                self.backbone_ckpt = backbone_ckpt
                self.student_model = ''

        defense_args = DefenseArgs(
            ckpt=hyperlora_ckpt_path,
            seed_pool_size=seed_pool_size,
            dataset=self.dataset_name,
            model=self.model_arch,
            backbone_ckpt=os.path.join(self.model_dir, 'checkpoint.pth.tar')
        )

        # 1. 创建 DynamicAttentionModel 实例。它内部会加载正确的骨干网络权重。
        self.defense_module = DynamicAttentionModel(defense_args, num_classes=self.num_classes)
        
        # 2. **修正点**: 调用新的方法，只加载防御相关的权重，而不是整个 state_dict。
        defense_state_dict = torch.load(hyperlora_ckpt_path, map_location='cpu', weights_only=False)
        self.defense_module.load_defense_modules(defense_state_dict)
        
        self.defense_module.to(self.device)
        self.defense_module.eval()

        # 初始化种子策略
        self.samples_per_seed = samples_per_seed
        self.seed_pool_size = seed_pool_size
        self.last_seed = random.randint(0, self.seed_pool_size - 1)
        self.samples_with_current_seed = 0
        
        print("=> HyperLoRA_DS Defense Initialized and Ready.")

    @classmethod
    def from_modeldir(cls, model_dir, device=None, **kwargs):
        # ... (这部分代码无需修改，保持原样) ...
        if 'hyperlora_ckpt_path' not in kwargs:
            raise ValueError("HyperLoRA defense requires 'hyperlora_ckpt_path' argument.")
        
        hyperlora_ckpt_path = kwargs.pop('hyperlora_ckpt_path')
        seed_pool_size = kwargs.pop('seed_pool_size', 2048)
        samples_per_seed = kwargs.pop('samples_per_seed', 64)
        
        backbone_box = Blackbox.from_modeldir(model_dir, device=device, **kwargs)
        
        return cls(model=backbone_box.model, 
                   hyperlora_ckpt_path=hyperlora_ckpt_path,
                   seed_pool_size=seed_pool_size,
                   samples_per_seed=samples_per_seed,
                   device=device,
                   output_type=backbone_box.output_type,
                   dataset_name=backbone_box.dataset_name, 
                   modelfamily=backbone_box.modelfamily, 
                   model_arch=backbone_box.model_arch, 
                   num_classes=backbone_box.num_classes,
                   model_dir=backbone_box.model_dir, 
                   out_path=backbone_box.out_path, 
                   log_prefix=backbone_box.log_prefix)

    # ... ( _select_next_seeds 和 __call__ 方法保持不变) ...
    def _select_next_seeds(self, batch_size, device):
        next_seeds = []
        for _ in range(batch_size):
            if self.samples_with_current_seed >= self.samples_per_seed:
                self.samples_with_current_seed = 0
                self.last_seed = random.randint(0, self.seed_pool_size - 1)
            next_seeds.append(self.last_seed)
            self.samples_with_current_seed += 1
        return torch.tensor(next_seeds, device=device, dtype=torch.long)

    def __call__(self, x, stat=True, return_origin=False):
        with torch.no_grad():
            seeds_to_use = self._select_next_seeds(x.shape[0], self.device)
            logits, _ = self.defense_module(x, seeds_to_use)
            y_prime = F.softmax(logits, dim=1)

        if stat and self.log_path:
            with torch.no_grad():
                y_v_logits = self.model(x)
                y_v = F.softmax(y_v_logits, dim=1)

            self.call_count += x.shape[0]
            self.queries.append((y_v.cpu().detach().numpy(), y_prime.cpu().detach().numpy()))

            if self.call_count > 0 and (self.call_count % 1000 < x.shape[0] or x.shape[0] >= 1000):
                query_out_path = osp.join(self.out_path, 'queries.pickle')
                with open(query_out_path, 'wb') as wf:
                    pickle.dump(self.queries, wf)
                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.calc_query_distances(self.queries)
                with open(self.log_path, 'a') as af:
                    test_cols = [self.call_count, l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        if return_origin:
            if 'y_v' not in locals():
                with torch.no_grad():
                    y_v_logits = self.model(x)
                    y_v = F.softmax(y_v_logits, dim=1)
            return y_prime, y_v
        else:
            return y_prime