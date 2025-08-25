# defenses/victim/train_hyperlora.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import sys
import json
import numpy as np

# --- 模块导入 ---
# 确保在项目根目录下运行，以便下面的导入能正常工作
sys.path.append(os.getcwd())
from defenses.victim.utils.hyperlora_utils import DynamicAttentionModel
from defenses.victim.utils.my_utils import get_classifier
from defenses import datasets
from defenses.victim.utils.domain_shift_utils import get_noise_level, add_gaussian_noise

# --- 梯度操作辅助函数 ---
def _flatten_grads(grads):
    valid_grads = [g for g in grads if g is not None]
    if not valid_grads:
        device = 'cpu'
        # 尝试从参数列表中找到设备
        for grad_list in grads:
            if isinstance(grad_list, (list, tuple)):
                for p in grad_list:
                    if hasattr(p, 'device'):
                        device = p.device
                        break
            if device != 'cpu':
                break
        return torch.tensor([], device=device)
    return torch.cat([g.flatten() for g in valid_grads])

def _unflatten_grads(flat_grad, params_with_grads):
    offset = 0
    unflattened = []
    # params_with_grads 应该是原始梯度列表，用于提供形状和None的位置
    for p in params_with_grads:
        if p is not None:
            numel = p.numel()
            unflattened.append(flat_grad[offset:offset+numel].view_as(p))
            offset += numel
        else:
            unflattened.append(None)
    return unflattened

def apply_grad_pcgrad(g_expert, g_task):
    g_expert_flat = _flatten_grads(g_expert)
    g_task_flat = _flatten_grads(g_task)
    
    if g_task_flat.numel() == 0 or g_expert_flat.numel() == 0: # 如果任一梯度为空，则无需操作
        return g_expert, g_task

    dot_product = torch.dot(g_expert_flat, g_task_flat)
    if dot_product < 0:
        # 投影任务梯度
        g_task_flat_proj = g_task_flat - (dot_product / torch.dot(g_expert_flat, g_expert_flat).clamp(min=1e-8)) * g_expert_flat
        g_task_proj = _unflatten_grads(g_task_flat_proj, g_task)
        return g_expert, g_task_proj
    else:
        return g_expert, g_task

# --- 参数解析 ---
parser = argparse.ArgumentParser(description='Advanced HyperLoRA Training for ModelGuard Integration')
parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10','CIFAR100', 'Caltech256', 'CUBS200'])
parser.add_argument('--model', type=str, required=True, help="Backbone model arch.")
parser.add_argument('--backbone_ckpt', type=str, required=True, help="Path to UNDEFENDED backbone weights.")
parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained defense module.")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--seed_pool_size', type=int, default=2048)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr_acc', type=float, default=1e-4)
parser.add_argument('--lr_div', type=float, default=1e-4)
parser.add_argument('--lr_basis', type=float, default=1e-5)
parser.add_argument('--lambda_acc', type=float, default=1.0, help="Weight for accuracy expert's gradient on basis.")
parser.add_argument('--lambda_div', type=float, default=0.5, help="Weight for diversity term in diversity task loss.")
parser.add_argument('--lambda_acc_penalty', type=float, default=0.1, help="Weight for accuracy penalty in diversity task loss.")
parser.add_argument('--grad_strategy', type=str, default='pcgrad', choices=['none', 'pcgrad'])
parser.add_argument('--noise_schedule', type=str, default='small_to_large', choices=['fixed', 'large_to_small', 'small_to_large'])
parser.add_argument('--min_noise_level', type=float, default=0.0)
parser.add_argument('--max_noise_level', type=float, default=0.08)
parser.add_argument('--fixed_noise_level', type=float, default=0.05)
parser.add_argument('--student_model', type=str, default='resnet18_8x') # Dummy arg, required by my_utils
args = parser.parse_args()

# --- 全局设置与数据加载 ---
acc, acc_best = 0, 0
dataset_class = datasets.__dict__[args.dataset]
modelfamily = datasets.dataset_to_modelfamily[args.dataset]

# 动态确定类别数
if args.dataset == 'CIFAR10': num_classes = 10
elif args.dataset == 'CIFAR100': num_classes = 100
elif args.dataset == 'Caltech256': num_classes = 256
elif args.dataset == 'CUBS200': num_classes = 200
else: raise ValueError(f"Unknown dataset for num_classes: {args.dataset}")
args.num_classes = num_classes

print("--- Loading Data ---")
train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']

data_train = dataset_class(train=True, transform=train_transform, download=True)
data_test = dataset_class(train=False, transform=test_transform, download=True)

data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=4, pin_memory=True)
print(f"Dataset '{args.dataset}' loaded from project's default data directory.")

# --- 模型、损失函数和优化器 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using Device: {device} ---")
net = DynamicAttentionModel(args, num_classes=num_classes).to(device)
criterion_ce = torch.nn.CrossEntropyLoss().to(device)

accuracy_params = [p for p in net.parameters() if hasattr(p, 'param_group') and p.param_group == 'accuracy']
diversity_params = [p for p in net.parameters() if hasattr(p, 'param_group') and p.param_group == 'diversity']
basis_params = [p for p in net.parameters() if hasattr(p, 'param_group') and p.param_group == 'basis']

print(f"Optimizing {sum(p.numel() for p in accuracy_params)} Acc params, {sum(p.numel() for p in diversity_params)} Div params, {sum(p.numel() for p in basis_params)} Basis params.")

optimizer_acc = torch.optim.Adam(accuracy_params, lr=args.lr_acc, weight_decay=1e-5)
optimizer_div = torch.optim.Adam(diversity_params, lr=args.lr_div, weight_decay=1e-5)
optimizer_basis = torch.optim.Adam(basis_params, lr=args.lr_basis, weight_decay=1e-5)

scheduler_acc = torch.optim.lr_scheduler.MultiStepLR(optimizer_acc, milestones=[100, 150], gamma=0.1)
scheduler_div = torch.optim.lr_scheduler.MultiStepLR(optimizer_div, milestones=[100, 150], gamma=0.1)
scheduler_basis = torch.optim.lr_scheduler.MultiStepLR(optimizer_basis, milestones=[100, 150], gamma=0.1)

# --- 训练函数 ---
def train(epoch):
    net.train()
    total_loss_acc, total_loss_div_task = 0, 0
    current_sigma = get_noise_level(args.noise_schedule, epoch, args.epochs, args.min_noise_level, args.max_noise_level, args.fixed_noise_level)
    progress_bar = tqdm(data_train_loader, desc=f"Epoch {epoch}/{args.epochs} [Grad: {args.grad_strategy}, Noise σ={current_sigma:.4f}]")

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # --- 步骤 A: Acc网络和共享基底的联合更新 ---
        optimizer_acc.zero_grad()
        optimizer_basis.zero_grad()
        
        seeds_acc = torch.randint(0, args.seed_pool_size, (images.shape[0],), device=device)
        logits_acc_clean, _ = net.forward_acc_only(images, seeds_acc)
        L_acc_clean = criterion_ce(logits_acc_clean, labels)
        
        L_acc_clean.backward()
        optimizer_acc.step()
        optimizer_basis.step()
        
        # --- 步骤 B: Div网络和共享基底的专家监督更新 ---
        optimizer_div.zero_grad()
        optimizer_basis.zero_grad()
        
        noisy_images = add_gaussian_noise(images, sigma=current_sigma)
        seeds_div = torch.randint(0, args.seed_pool_size, (noisy_images.shape[0],), device=device)

        logits_expert_on_noisy, _ = net.forward_acc_only(noisy_images, seeds_div)
        L_expert_acc = criterion_ce(logits_expert_on_noisy, labels)
        g_expert_acc_on_basis = torch.autograd.grad(L_expert_acc, basis_params, retain_graph=True, allow_unused=True)

        logits_div_on_noisy, features_div_on_noisy = net(noisy_images, seeds_div)
        L_acc_penalty = criterion_ce(logits_div_on_noisy, labels)
        
        z_embed = net.seed_embedding(seeds_div)
        z_embed.requires_grad_(True)
        _, features_for_grad = net.forward_from_z_embed(net.feature_extractor(noisy_images), z_embed)
        target_tensor = F.normalize(features_for_grad, p=2, dim=1)
        sum_target = target_tensor.sum()
        grad_target = torch.autograd.grad(sum_target, z_embed, torch.ones_like(sum_target), create_graph=True)[0]
        L_diversity_term = - (grad_target.norm(p=2, dim=1)**2).mean()
        
        L_div_task = args.lambda_div * L_diversity_term + args.lambda_acc_penalty * L_acc_penalty
        
        g_div_task_on_basis = torch.autograd.grad(L_div_task, basis_params, retain_graph=True, allow_unused=True)
        g_div_on_self = torch.autograd.grad(L_div_task, diversity_params, allow_unused=True)

        if args.grad_strategy == 'pcgrad':
            g_expert_final, g_div_final_on_basis = apply_grad_pcgrad(g_expert_acc_on_basis, g_div_task_on_basis)
        else:
            g_expert_final, g_div_final_on_basis = g_expert_acc_on_basis, g_div_task_on_basis
        
        for p, g in zip(diversity_params, g_div_on_self):
            if g is not None: p.grad = g
        optimizer_div.step()

        final_grad_on_basis = []
        for ga, gd in zip(g_expert_final, g_div_final_on_basis):
             if ga is not None and gd is not None:
                 final_grad_on_basis.append(args.lambda_acc * ga + args.lambda_div * gd)
             elif ga is not None:
                 final_grad_on_basis.append(args.lambda_acc * ga)
             elif gd is not None:
                 final_grad_on_basis.append(args.lambda_div * gd)
             else:
                 final_grad_on_basis.append(None)
        
        for p, g in zip(basis_params, final_grad_on_basis):
            if g is not None: p.grad = g
        optimizer_basis.step()

        total_loss_acc += L_acc_clean.item()
        total_loss_div_task += L_div_task.item()
        progress_bar.set_postfix({'Acc Loss': f"{total_loss_acc/(i+1):.4f}", 'Div Task Loss': f"{total_loss_div_task/(i+1):.4f}"})

# --- 测试函数 ---
def test():
    global acc, acc_best
    net.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in data_test_loader:
            images, labels = images.to(device), labels.to(device)
            seeds_test = torch.randint(0, args.seed_pool_size, (images.shape[0],), device=device)
            output, _ = net(images, seeds_test)
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data).sum().item()
            total_samples += images.shape[0]

    acc = float(total_correct) / total_samples
    is_best = acc > acc_best
    if is_best: acc_best = acc
    print(f'\nTest Accuracy: {acc:.4f} (Best: {acc_best:.4f}) {"*** NEW BEST ***" if is_best else ""}')
    return acc, is_best

# --- 主执行流程 ---
if __name__ == '__main__':
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models and logs will be saved to: {save_dir}")
    
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    for e in range(1, args.epochs + 1):
        train(e)
        current_acc, is_best = test()
        scheduler_acc.step()
        scheduler_div.step()
        scheduler_basis.step()
        if is_best:
            save_path = os.path.join(save_dir, 'model_best.pth')
            torch.save(net.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    print(f"\nTraining finished! Best test accuracy achieved: {acc_best:.4f}")