import torch
import os
import json
from torchvision.utils import save_image

class PatchAttack_save:
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
        
        return adv_images, images, msk, target

    def save_results(self, adv_images, images, msk, target):
        os.makedirs('./adv_examples2007/train/cleans', exist_ok=True)
        os.makedirs('./adv_examples2007/train/images', exist_ok=True)
        os.makedirs('./adv_examples2007/train/labels', exist_ok=True)
        os.makedirs('./adv_examples2007/train/masks', exist_ok=True)

        labels = []
        for i in range(len(images)):
            global a
            clean_filename = f'image_{a}.png'
            image_filename = f'adv_patch_{a}.png'
            mask_filename = f'mask_{a}.png'
            save_image(images[i], f'./adv_examples2007/train/cleans/{clean_filename}')
            save_image(adv_images[i], f'./adv_examples2007/train/images/{image_filename}')
            save_image(msk[i], f'./adv_examples2007/train/masks/{mask_filename}')
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
            a += 1

        with open('./adv_examples2007/train/labels/labels.json', 'a') as label_file:
            json.dump(labels, label_file, indent=4)
            
class PatchAttack_patchZero:
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
        self.num_classes = 2
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
        images = input
        b, c, h, w = images.shape
        msk = images.new_zeros(images.shape).cuda()
        b, c, h, w = images.shape
        if self.position == 'uniform':
            x = torch.randint(0, int(w - self.patch_size), (1,))
            y = torch.randint(0, int(h - self.patch_size), (1,))
            x = x.item()
            y = y.item()
        msk[:, :, y:y+int(self.patch_size), x:x+int(self.patch_size)] = 1
        msk = msk.long()
        msk_class_pre = torch.ones(b, self.num_classes, dtype=torch.float).cuda()
        adv_images = images.clone().cuda()
        
        for _ in range(self.steps):
            adv_images.requires_grad = True

            loss = self.model(adv_images, msk, msk_class_pre, target)
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images + self.step_size * grad.sign() * msk
            msk_class_pre[:, 0]=1
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        return adv_images, images, msk, target