import os
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
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
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): One of 'train', 'test', 'val'.
            num_classes (int): Total number of classes in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_classes = 2
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.cleans_dir = os.path.join(root_dir, split, 'cleans')
        self.masks_dir = os.path.join(root_dir, split, 'masks')
        self.labels_path = os.path.join(root_dir, split, 'labels', 'labels_.json')

        with open(self.labels_path, 'r') as file:
            self.labels = json.load(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_info = self.labels[idx]
        image_name = label_info['filename']
        clean_name = label_info['clean']
        mask_name = label_info['mask']

        if random.random() > 0.5:
            image_path = os.path.join(self.images_dir, image_name)
            mask_path = os.path.join(self.masks_dir, mask_name)
        else:
            image_path = os.path.join(self.cleans_dir, clean_name)
            mask_path = None

        image = Image.open(image_path).convert('RGB')
        
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask[mask > 0] = 1  # 将非黑色像素值设为1
            mask = torch.tensor(mask, dtype=torch.int64)
            class_pre = torch.ones(self.num_classes, dtype=torch.float)
        else:
            mask = torch.zeros((image.size[1], image.size[0]), dtype=torch.int64)
            class_pre = torch.zeros(self.num_classes, dtype=torch.float)
            class_pre[0]=1
        

        if self.transform:
            image = self.transform(image)

        return image, mask, class_pre

def save_mask_as_image(mask, file_path):

    mask = mask * 255
    mask = mask.byte()
    transform = transforms.ToPILImage()
    mask_image = transform(mask)
    mask_image.save(file_path)


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


@click.command()
@click.option('--data-path', type=str, default='/home/huangyifan/ssd/ICLR_rebuttal/patchZero/adv_examples2007', help='Path to dataset folder')
@click.option('--models-path', type=str,default='/home/huangyifan/ssd/ICLR_rebuttal/patchZero/pspnet', help='Path for storing model snapshots')
@click.option('--backend', type=str, default='resnet50', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=16)
@click.option('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='4', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.0001)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
def eval(data_path, models_path, backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, milestones, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = build_network('/home/huangyifan/ssd/ICLR_rebuttal/patchZero/pspnet/PSPNet_5', backend)
    data_path = os.path.abspath(os.path.expanduser(data_path))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = CustomDataset(root_dir='/home/huangyifan/ssd/ICLR_rebuttal/patchZero/adv_examples2007', split='test', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    net.train()
    for x, y, y_cls in tqdm(dataloader, total=len(dataset) // batch_size + 1):
        save_mask_as_image(x.squeeze(0), 'output_stage1_test/image.png')
        x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()
        out, out_cls = net(x)
        out = F.interpolate(out, size=y.shape[1:], mode='bilinear', align_corners=False)
        mask = torch.sigmoid(out)
        save_mask_as_image(mask.squeeze(0), 'output_stage1_test/mask.png')
        exit()
        



if __name__ == '__main__':
    eval()
