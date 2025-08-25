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
from defenses.victim.utils.hyperlora_utils_single_network import DynamicAttentionModel
from defenses.victim.utils.my_utils import get_classifier
from defenses import datasets
from defenses.victim.utils.domain_shift_utils import get_noise_level, add_gaussian_noise

# --- 梯度操作辅助函数 ---
def _flatten_grads(grads):
    """将梯度列表展平为一个向量。"""
    valid_grads = [g for g in grads if g is not None]
    if not valid_grads:
        # 如果没有有效的梯度，返回一个空的tensor
        device = 'cpu'
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
    """将展平的梯度向量恢复为原始参数的形状。"""
    offset = 0
    unflattened = []
    for p in params_with_grads:
        if p is not None:
            numel = p.numel()
            unflattened.append(flat_grad[offset:offset+numel].view_as(p))
            offset += numel
        else:
            unflattened.append(None)
    return unflattened

def apply_grad_pcgrad(g_expert, g_task, batch_idx, epoch):
    """应用PCGrad策略，并加入诊断打印"""
    g_expert_flat = _flatten_grads(g_expert)
    g_task_flat = _flatten_grads(g_task)
    
    if g_task_flat.numel() == 0 or g_expert_flat.numel() == 0:
        return g_expert, g_task

    dot_product = torch.dot(g_expert_flat, g_task_flat)
    
    # 【诊断点】打印梯度相似度
    #if batch_idx % 100 == 0:
    #    norm_expert = torch.norm(g_expert_flat)
    #    norm_task = torch.norm(g_task_flat)
    #    cosine_sim = dot_product / (norm_expert * norm_task + 1e-8)
    #    print(f"\n[DEBUG E{epoch}:B{batch_idx}] PCGrad Dot: {dot_product.item():.4f}, Cosine Sim: {cosine_sim.item():.4f}")

    if dot_product < 0:
        #if batch_idx % 100 == 0:
            #print(f"[DEBUG E{epoch}:B{batch_idx}] Gradient Conflict DETECTED. Projecting task gradient.")
        g_task_flat_proj = g_task_flat - (dot_product / torch.dot(g_expert_flat, g_expert_flat).clamp(min=1e-8)) * g_expert_flat
        g_task_proj = _unflatten_grads(g_task_flat_proj, g_task)
        return g_expert, g_task_proj
    else:
        return g_expert, g_task

# --- 参数解析 ---
parser = argparse.ArgumentParser(description='Advanced HyperLoRA Training for ModelGuard Integration')
parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10','CIFAR100', 'Caltech256', 'CUBS200'], help="Dataset to use for training.")
parser.add_argument('--model', type=str, required=True, help="Backbone model architecture (e.g., vgg16_bn, resnet50).")
parser.add_argument('--backbone_ckpt', type=str, required=True, help="Path to the checkpoint of the undefended backbone model.")
parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained defense module and logs.")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
parser.add_argument('--z_dim', type=int, default=64, help="Dimension of the seed embedding vector.")
parser.add_argument('--seed_pool_size', type=int, default=2048, help="Size of the random seed pool.")
parser.add_argument('--epochs', type=int, default=200, help="Total number of training epochs.")
parser.add_argument('--lr_hypernet', type=float, default=1e-4, help="Learning rate for the single hypernetwork and seed embedding.")
parser.add_argument('--lr_basis', type=float, default=2e-5, help="Learning rate for the shared LoRA basis.")
parser.add_argument('--lr_acc', type=float, default=1e-4, help="Learning rate for the accuracy hypernetwork.")
parser.add_argument('--lr_div', type=float, default=1e-4, help="Learning rate for the diversity hypernetwork and seed embedding.")
parser.add_argument('--lambda_acc', type=float, default=1.0, help="Weight for accuracy expert's gradient on the shared basis.")
parser.add_argument('--lambda_div', type=float, default=0.1, help="Weight for the diversity term in the diversity task loss.")
parser.add_argument('--lambda_acc_penalty', type=float, default=0.1, help="Weight for the accuracy penalty in the diversity task loss.")
parser.add_argument('--grad_strategy', type=str, default='pcgrad', choices=['none', 'pcgrad'], help="Strategy for handling conflicting gradients.")
parser.add_argument('--noise_schedule', type=str, default='small_to_large', choices=['fixed', 'large_to_small', 'small_to_large'], help="Schedule for applying Gaussian noise.")
parser.add_argument('--min_noise_level', type=float, default=0.0, help="Minimum sigma for Gaussian noise.")
parser.add_argument('--max_noise_level', type=float, default=0.08, help="Maximum sigma for Gaussian noise.")
parser.add_argument('--fixed_noise_level', type=float, default=0.05, help="Fixed sigma for Gaussian noise if schedule is 'fixed'.")
parser.add_argument('--grad_clip_norm', type=float, default=1.0, help="Maximum norm for gradient clipping.")
parser.add_argument('--student_model', type=str, default='resnet18_8x', help="Dummy argument required by my_utils, not used.") # 兼容性参数
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

# --- 模型、损失函数和优化器 (核心修改) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DynamicAttentionModel(args, num_classes=num_classes).to(device)
criterion_ce = torch.nn.CrossEntropyLoss().to(device)

# 新的参数分组
hypernet_params = [p for p in net.parameters() if hasattr(p, 'param_group') and p.param_group == 'hypernet']
basis_params = [p for p in net.parameters() if hasattr(p, 'param_group') and p.param_group == 'basis']

print(f"Optimizing {sum(p.numel() for p in hypernet_params)} HyperNet params, {sum(p.numel() for p in basis_params)} Basis params.")

optimizer_hypernet = torch.optim.Adam(hypernet_params, lr=args.lr_hypernet, weight_decay=1e-5)
optimizer_basis = torch.optim.Adam(basis_params, lr=args.lr_basis, weight_decay=1e-5)

scheduler_hypernet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hypernet, T_max=args.epochs)
scheduler_basis = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_basis, T_max=args.epochs)

# --- 训练函数 ---
def train(epoch):
    net.train()
    total_loss_acc, total_loss_div = 0, 0
    current_sigma = get_noise_level(args.noise_schedule, epoch, args.epochs, args.min_noise_level, args.max_noise_level, args.fixed_noise_level)
    progress_bar = tqdm(data_train_loader, desc=f"Epoch {epoch}/{args.epochs} [PCGrad on SingleNet, Noise σ={current_sigma:.4f}]")

    # 将所有可训练参数合并，方便后续处理
    trainable_params = hypernet_params + basis_params

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # --- 梯度清零 ---
        optimizer_hypernet.zero_grad()
        optimizer_basis.zero_grad()
        
        # --- 任务1: 准确性 (专家任务) ---
        # 计算在干净数据上的准确性损失
        seeds_acc = torch.randint(0, args.seed_pool_size, (images.shape[0],), device=device)
        logits_acc, _ = net(images, seeds_acc)
        L_acc = criterion_ce(logits_acc, labels)
        
        # 计算 L_acc 关于所有可训练参数的梯度
        g_acc = torch.autograd.grad(L_acc, trainable_params, retain_graph=True, allow_unused=True)

        # --- 任务2: 多样性 (辅助任务) ---
        # 准备加噪数据
        noisy_images = add_gaussian_noise(images, sigma=current_sigma)
        seeds_div = torch.randint(0, args.seed_pool_size, (noisy_images.shape[0],), device=device)

        # 归一化 z_embed
        z_embed_raw = net.seed_embedding(seeds_div)
        z_embed = F.normalize(z_embed_raw, p=2, dim=1).requires_grad_(True)
        
        # 计算多样性损失
        logits_div, features_for_grad = net.forward_from_z_embed(net.feature_extractor(noisy_images), z_embed)
        
        # 确保batch size为偶数以使用余弦相似度
        if features_for_grad.shape[0] % 2 != 0:
             features_for_grad = features_for_grad[:-1]
        
        features1, features2 = torch.chunk(features_for_grad, 2, dim=0)
        cosine_sim = F.cosine_similarity(features1, features2, dim=1)
        L_div = cosine_sim.mean() # 目标是最小化相似度
        
        # 计算 L_div 关于所有可训练参数的梯度
        g_div = torch.autograd.grad(L_div, trainable_params, allow_unused=True)
        
        # --- 梯度融合与更新 ---
        
        # 使用PCGrad处理梯度冲突
        g_acc_final, g_div_final = apply_grad_pcgrad(g_acc, g_div, i, epoch)
        
        # 将最终的组合梯度赋给模型参数
        # final_grad = lambda_acc * g_acc_final + lambda_div * g_div_final
        final_grad = []
        for ga, gd in zip(g_acc_final, g_div_final):
            # 必须检查None，因为某些参数可能只与一个任务相关
            if ga is not None and gd is not None:
                final_grad.append(args.lambda_acc * ga + args.lambda_div * gd)
            elif ga is not None:
                final_grad.append(args.lambda_acc * ga)
            elif gd is not None:
                final_grad.append(args.lambda_div * gd)
            else:
                final_grad.append(None)

        # 将计算好的梯度手动赋给.grad属性
        for p, g in zip(trainable_params, final_grad):
            if g is not None:
                p.grad = g

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
        
        # 执行更新
        optimizer_hypernet.step()
        optimizer_basis.step()

        total_loss_acc += L_acc.item()
        total_loss_div += L_div.item()
        progress_bar.set_postfix({'Acc Loss': f"{total_loss_acc/(i+1):.4f}", 'Div Loss': f"{total_loss_div/(i+1):.4f}"})

# --- 测试函数 ---
def test():
    global acc, acc_best
    net.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in data_test_loader:
            images, labels = images.to(device), labels.to(device)
            # 在测试时，使用完整的、结合了acc和div的扰动
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
    
    # 保存训练参数
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    for e in range(1, args.epochs + 1):
        train(e)
        current_acc, is_best = test()
        
        scheduler_hypernet.step()
        scheduler_basis.step()
        
        if is_best:
            save_path = os.path.join(save_dir, 'model_best.pth')
            # 只保存模型的状态字典
            torch.save(net.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    print(f"\nTraining finished! Best test accuracy achieved: {acc_best:.4f}")