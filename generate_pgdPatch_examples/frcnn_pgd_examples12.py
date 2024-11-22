
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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

a = 0

class PatchAttack:
    def __init__(self, model, patch_size, step_size, eps, steps, position='uniform', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        '''
        model: nn.Module, the model to attack
        patch_size: float, the ratio of the patch length to the image length
        step_size: float, the step size of the attack
        eps: float, the bound of the patch perturbation
        steps: int, the number of attack steps
        position: str, the position of the patch, 'uniform' or 'center'
        '''
        self.model = model
        self.patch_size = patch_size
        self.step_size = step_size
        self.eps = eps
        self.steps = steps
        self.position = position
        self.mean = mean
        self.std = std
        if mean is not None:
            self.mean = torch.Tensor(mean).view(1, 3, 1, 1)
            self.std = torch.Tensor(std).view(1, 3, 1, 1)
        
    
    def inverse_normalize(self, input):
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)
        return input * self.std + self.mean
    
    def normalize(self, input):
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)
        return (input - self.mean) / self.std

    def attack(self, input, target):
        images = torch.stack(input)
        msk = images.new_zeros(images.shape).to(device)
        b, c, h, w = images.shape
        if self.position == 'uniform':
            x = torch.randint(0, int(w - self.patch_size), (1,))
            y = torch.randint(0, int(h - self.patch_size), (1,))
            x = x.item()
            y = y.item()
        msk[:, :, y:y+int(self.patch_size), x:x+int(self.patch_size)] = 1
        adv_images = images.clone().to(device)
        for _ in range(self.steps):
            adv_images.requires_grad = True

            loss_dict = self.model(adv_images, target)
            loss = sum(loss for loss in loss_dict.values())
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images + self.step_size * grad.sign() * msk
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        os.makedirs('./adv_examples2012/train/cleans', exist_ok=True)    
        os.makedirs('./adv_examples2012/train/images', exist_ok=True)
        os.makedirs('./adv_examples2012/train/labels', exist_ok=True)
        os.makedirs('./adv_examples2012/train/masks', exist_ok=True)

            # 保存每个图像和对应的标签
        labels = []
        for i in range(b):
            global a
            clean_filename = f'image_{a}.png'
            image_filename = f'adv_patch_{a}.png'
            mask_filename = f'mask_{a}.png'
            save_image(images[i], f'./adv_examples2012/train/cleans/{clean_filename}')
            save_image(adv_images[i], f'./adv_examples2012/train/images/{image_filename}')
            save_image(msk[i], f'./adv_examples2012/train/masks/{mask_filename}')
            label_info = {
                'filename': image_filename,
                'clean': clean_filename,
                'mask': mask_filename,
                'boxes': target[i]['boxes'].tolist(),
                'labels': target[i]['labels'].tolist(),
                'image_id': target[i]['image_id'].item(),
                'area': target[i]['area'].tolist(),
                'iscrowd': target[i]['iscrowd'].tolist()
            }
            labels.append(label_info)
            a += 1  # 增加全局变量 i

        # 将标签信息保存到一个JSON文件中
        with open('./adv_examples2012/train/labels/labels.json', 'a') as label_file:
            json.dump(labels, label_file, indent=4)
        return adv_images

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
    #dataset = VOCDetection('/gpfs/hulab/huangyifan/datasets', year='2007', image_set='train', download=False, transform=data_augmentation)
    dataset= VOCDetection('/gpfs/hulab/huangyifan/datasets', year='2012', image_set='train', download=False, transform=data_augmentation)
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
        output = atk.attack(images , targets)
    
if __name__ == '__main__':
    main()