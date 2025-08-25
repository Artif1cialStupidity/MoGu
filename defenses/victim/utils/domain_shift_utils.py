# defenses/victim/utils/domain_shift_utils.py

import torch
import random

def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    向一批图像张量中添加高斯噪声。

    Args:
        images (torch.Tensor): 输入的图像张量，通常是已经归一化的。
        sigma (float): 高斯噪声的标准差。

    Returns:
        torch.Tensor: 添加了噪声的图像张量。
    """
    if sigma <= 0:
        return images
    noise = torch.randn_like(images) * sigma
    noisy_images = images + noise
    # 注意：不需要裁剪(clamp)，因为我们是在归一化空间中操作。
    return noisy_images


def get_noise_level(schedule_type: str, epoch: int, total_epochs: int,
                    min_sigma: float, max_sigma: float, fixed_sigma: float) -> float:
    """
    根据指定的策略计算当前epoch的噪声水平(sigma)。

    Args:
        schedule_type (str): 噪声调度策略 ('fixed', 'large_to_small', etc.)
        epoch (int): 当前的epoch号 (从1开始)。
        total_epochs (int): 总的训练epoch数。
        min_sigma (float): 噪声标准差的下界。
        max_sigma (float): 噪声标准差的上界。
        fixed_sigma (float): 当策略为'fixed'时使用的固定噪声值。

    Returns:
        float: 当前epoch应使用的高斯噪声标准差。
    """
    
    # 进度比例 (从 0 到 1)
    # 我们用 epoch-1 使得第一个epoch的progress为0，最后一个epoch为 (total-1)/total
    progress = (epoch - 1) / (total_epochs - 1) if total_epochs > 1 else 0

    if schedule_type == 'fixed':
        return fixed_sigma

    elif schedule_type == 'large_to_small':
        # 从最大噪声线性下降到最小噪声
        sigma = max_sigma - (max_sigma - min_sigma) * progress
        return sigma

    elif schedule_type == 'small_to_large':
        # 从最小噪声线性增长到最大噪声 (课程学习)
        sigma = min_sigma + (max_sigma - min_sigma) * progress
        return sigma

    elif schedule_type == 'random_from_middle':
        # 从一个中心点开始，采样范围随epoch线性扩大
        mid_sigma = (min_sigma + max_sigma) / 2.0
        total_range = max_sigma - min_sigma
        
        # 当前允许的采样范围宽度
        current_range_width = total_range * progress
        
        # 计算当前采样区间的上下界
        lower_bound = max(min_sigma, mid_sigma - current_range_width / 2.0)
        upper_bound = min(max_sigma, mid_sigma + current_range_width / 2.0)
        
        # 在当前区间内均匀随机采样
        return random.uniform(lower_bound, upper_bound)

    else:
        raise ValueError(f"未知的噪声调度策略: {schedule_type}")
