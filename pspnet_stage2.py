import os
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import click
import json
import numpy as np
import random
from pspnet.pspnet import PSPNet
import torch.nn.functional as F
from pspDataset import CustomDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from attack import PatchAttack_patchZero
from torch.utils.data import ConcatDataset

class PatchZero(nn.Module):
    def __init__(self, net, alpha, frcnn_path, num_classes=21):
        super(PatchZero, self).__init__()
        self.net = net
        self.frcnn = self._initialize_frcnn(frcnn_path, num_classes)
        self.alpha = alpha
        self.seg_criterion = nn.NLLLoss2d(weight=torch.tensor([1.0399, 26.0417]).cuda())
        self.cls_criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0399, 26.0417]).cuda())
        
    def _initialize_frcnn(self, frcnn_path, num_classes):
        # Initialize Faster R-CNN
        frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
        frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        frcnn.load_state_dict(torch.load(frcnn_path))
        return frcnn.cuda()

    def forward(self, x, mask, y_cls, targets):
        mask = mask[:, 0, :, :]
        out, out_cls = self.net(x)
        out = torch.softmax(out, dim=1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        with torch.no_grad():
            out_mask = torch.zeros(out.shape[0], 1, out.shape[2], out.shape[3], device=out.device)
            patch_condition = out[:, 0, :, :] > 0.5
            out_mask[:, 0, :, :] = patch_condition
        if mask.dtype != torch.long:
            mask = mask.long()
        out_mask = out_mask.detach() + torch.sigmoid(out[:, 0:1, :, :]) - out_mask.detach()
        inputs = out_mask * x
        frcnn_output = self.frcnn(inputs, targets)
        frcnn_loss = sum(loss for loss in frcnn_output.values())
        seg_loss = self.seg_criterion(out, mask)
        cls_loss = self.cls_criterion(out_cls, y_cls)
        net_loss = seg_loss + self.alpha * cls_loss
        
        return frcnn_loss + net_loss
    
    def train(self, mode=True):
        super(PatchZero, self).train(mode)
        self.net.train(mode)
        self.frcnn.train(mode)
        
def collate_fn(batch):
    """
    自定义 collate 函数，处理 targets 中 variable-sized 数据。
    """
    x = []
    mask = []
    y_cls = []
    targets = []
    
    for sample in batch:
        x.append(sample[0])  # 图像张量
        mask.append(sample[1])  # Mask 张量
        y_cls.append(sample[2])  # 分类标签张量
        targets.append(sample[3])  # 边界框和标签
    
    # 将大小一致的部分堆叠成张量
    x = torch.stack(x, dim=0)
    mask = torch.stack(mask, dim=0)
    y_cls = torch.stack(y_cls, dim=0)
    
    # 将 targets 保留为列表
    return x, mask, y_cls, targets

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=2, psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
    net = net.cuda()
    return net, epoch

def save_mask_as_image(mask, file_path):

    mask = mask * 255
    mask = mask.byte()
    transform = transforms.ToPILImage()
    mask_image = transform(mask)
    mask_image.save(file_path)


def train(batch_size, models_path, backend, snapshot, frcnn_path, start_lr, alpha, milestones, gpu, epochs):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = build_network(snapshot, backend)
    models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)
    
    patchZero = PatchZero(net, alpha, frcnn_path)
    patchZero.cuda()
    atk = PatchAttack_patchZero(patchZero, patch_size=120, step_size=0.1, eps=0.3, steps=100)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset7 = CustomDataset(root_dir='/home/huangyifan/ssd/ICLR_rebuttal/patchZero/adv_examples2007', split='train', transform=transform, mode = 'clean')
    dataset12 = CustomDataset(root_dir='/home/huangyifan/ssd/ICLR_rebuttal/patchZero/adv_examples2012', split='train', transform=transform, mode = 'clean')
    dataset = ConcatDataset([dataset7, dataset12])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    # train_loader, class_weights, n_images = None, None, None
    
    params = [p for p in patchZero.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=start_lr, weight_decay=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])
    
    for epoch in range(starting_epoch, starting_epoch + epochs):
        seg_criterion = nn.NLLLoss2d(weight=torch.tensor([1.0399, 26.0417]).cuda())
        cls_criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0399, 26.0417]).cuda())
        epoch_losses = []
        patchZero.train()
        for x, mask, y_cls,targets in tqdm(dataloader, total=len(dataset) // batch_size + 1):
            optimizer.zero_grad()
            x, mask, y_cls, targets = x.cuda(), mask.cuda(), y_cls.cuda(), [{k: v.cuda() for k, v in t.items()} for t in targets]
            adv_images, images, msk, target = atk.attack(x, targets)
            loss = patchZero(adv_images, msk, y_cls, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PatchZero", str(epoch + 1)])))
        train_loss = np.mean(epoch_losses)

        
if __name__ == '__main__':
    batch_size = 4
    models_path = '/home/huangyifan/ssd/ICLR_rebuttal/patchZero/pspnet/fineTune'
    backend = 'resnet50'
    snapshot = '/home/huangyifan/ssd/ICLR_rebuttal/patchZero/pspnet/PSPNet_5'
    frcnn_path = '/home/huangyifan/ssd/ICLR_rebuttal/patchZero/best_fasterrcnn.pth'
    start_lr = 0.0001
    alpha = 1.0
    milestones = '10,20,30'
    gpu = '0'
    epochs = 20
    train(batch_size, models_path, backend, snapshot, frcnn_path, start_lr, alpha, milestones, gpu, epochs)
