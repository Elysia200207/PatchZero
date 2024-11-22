
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
from torchvision.utils import save_image
import json
import torch
from attack import PatchAttack
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

a = 0

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


def main():
    data_augmentation = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(img_size),
                transforms.Resize((375, 500)),
                transforms.ToTensor()
                # transforms.Normalize(mean=[0.407, 0.457, 0.485],
                #                     std=[1,1,1])
                                    ])
            
    # 加载数据集
    dataset = VOCDetection('/gpfs/hulab/huangyifan/datasets', year='2007', image_set='train', download=False, transform=data_augmentation)
    # dataset_2012_train = VOCDetection('/gpfs/hulab/huangyifan/datasets', year='2012', image_set='train', download=False, transform=data_augmentation)
    # dataset = torch.utils.data.ConcatDataset([dataset_2007_train, dataset_2012_train])
    dataset_test = VOCDetection('/gpfs/hulab/huangyifan/datasets', year='2007', image_set='val', download=True, transform=data_augmentation)

    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))


    model_path = '/home/huangyifan/ssd/ICLR_rebuttal/patchZero/best_fasterrcnn.pth'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 21
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.train()

    print(3)


    model.to(device)
    atk = PatchAttack(model, patch_size=120, step_size=0.1, eps=0.3, steps=100)

    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"generate example:")):
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        targets = normalize_targets(targets, label_map, device)
        adv_images, images, msk, target = atk.attack(images , targets)
        atk.save_results(adv_images, images, msk, target)
    
if __name__ == '__main__':
    main()