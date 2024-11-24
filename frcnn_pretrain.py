import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
from torchvision.ops import box_iou
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch

label_map = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}

def normalize_targets(targets, label_map, device):
    normalized_targets = []

    for target in targets:
        annotation = target['annotation']
        objects = annotation['object']
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in objects:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_map[obj['name']])
            areas.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(int(obj.get('difficult', 0)))

        normalized_targets.append({
            'boxes': torch.tensor(boxes, dtype=torch.float32).to(device),
            'labels': torch.tensor(labels, dtype=torch.int64).to(device),
            'image_id': torch.tensor([int(annotation['filename'].split('.')[0].split('_')[-1])]).to(device),
            'area': torch.tensor(areas, dtype=torch.float32).to(device),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.int64).to(device)
        })

    return normalized_targets

transforms_=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
data_augmentation = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                std=[1,1,1])])
        


# 加载数据集
dataset_2007_train = VOCDetection('/path/to/VOCdataset', year='2007', image_set='train', download=False, transform=data_augmentation)
dataset_2012_train = VOCDetection('/path/to/VOCdataset', year='2012', image_set='train', download=False, transform=data_augmentation)
dataset = torch.utils.data.ConcatDataset([dataset_2007_train, dataset_2012_train])
dataset_test = VOCDetection('/path/to/VOCdataset', year='2007', image_set='val', download=True, transform=data_augmentation)

data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# 定义计算 AP50 的函数
def calculate_ap50(outputs, targets):
    ap50 = 0
    for output, target in zip(outputs, targets):
        pred_boxes = output['boxes']
        pred_scores = output['scores']
        true_boxes = target['boxes']

        if len(pred_boxes) == 0 or len(true_boxes) == 0:
            continue

        ious = box_iou(pred_boxes, true_boxes)
        max_iou, _ = ious.max(dim=1)
        ap50 += (max_iou >= 0.5).float().mean().item()
    
    return ap50 / len(outputs)

# 加载预训练模型并修改分类器
# model_path = '/path/to/pt/best_fasterrcnn.pth'

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#model.load_state_dict(torch.load(model_path))
# 定义优化器和学习率调度器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-6, weight_decay=0.0001)


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 创建 TensorBoard 记录器
writer = SummaryWriter()
print(f"TensorBoard logs are being saved to: {writer.log_dir}")

# 训练模型
num_epochs = 50
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

best_ap50 = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for i, (images, targets) in enumerate(progress_bar):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        targets = normalize_targets(targets, label_map, device)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())

        if i % 10 == 0:

            print(f"Epoch {epoch+1}, Step {i}, Loss: {losses.item()}")
            writer.add_scalar('Loss/train', losses.item(), epoch * len(data_loader) + i)
    #lr_scheduler.step()

    model.eval()
    ap50 = 0
    with torch.no_grad():
        for images, targets in tqdm(data_loader_test, desc="Validation"):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            targets = normalize_targets(targets, label_map, device)
            outputs = model(images)
            ap50 += calculate_ap50(outputs, targets)
    
    ap50 /= len(data_loader_test)
    print(f"Epoch: {epoch}, AP50: {ap50}")
    writer.add_scalar('AP50/val', ap50, epoch)
    
    if ap50 > best_ap50:
        best_ap50 = ap50
        best_epoch = epoch
        print("Replace checkpoint!")
        torch.save(model.state_dict(), "path/to/pt/best_fasterrcnn.pth")

print("Training complete!")