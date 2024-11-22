import torch

# 定义 mask 的尺寸
mask_height = 500
mask_width = 375

# 定义类别 1 的区域尺寸
region_height = 120
region_width = 120

# 计算类别 1 的像素数量
num_masks_with_region = 50
num_pixels_class_1 = num_masks_with_region * region_height * region_width

# 计算类别 0 的像素数量
num_masks_total = 100
num_pixels_total = num_masks_total * mask_height * mask_width
num_pixels_class_0 = num_pixels_total - num_pixels_class_1

# 计算类别频率
freq_class_0 = num_pixels_class_0 / num_pixels_total
freq_class_1 = num_pixels_class_1 / num_pixels_total

# 计算类别权重（频率的反比）
class_weights = torch.tensor([1.0 / freq_class_0, 1.0 / freq_class_1], dtype=torch.float)

print(class_weights)