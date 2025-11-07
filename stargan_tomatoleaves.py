# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as utils
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pickle
import csv
import math
import random
import os # Import os module
import time


# Custom Salt and Pepper Noise transform
class SaltAndPepperNoise(object):
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, img):
        img_np = np.array(img)
        row, col, ch = img_np.shape
        # Salt
        num_salt = np.ceil(self.amount * img_np.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_np.shape[:2]]
        img_np[coords[0], coords[1], :] = 255
        # Pepper
        num_pepper = np.ceil(self.amount * img_np.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_np.shape[:2]]
        img_np[coords[0], coords[1], :] = 0
        return transforms.ToPILImage()(img_np)

# Custom Occlusion transform
class Occlusion(object):
    def __init__(self, area_ratio):
        self.area_ratio = area_ratio

    def __call__(self, img):
        img_np = np.array(img)
        h, w, _ = img_np.shape
        mask_size = int(np.sqrt(self.area_ratio * h * w))
        x = np.random.randint(0, w - mask_size)
        y = np.random.randint(0, h - mask_size)
        img_np[y:y + mask_size, x:x + mask_size, :] = 0  # Black occlusion
        return transforms.ToPILImage()(img_np)
# <<< FIX: New class to replace the lambda function
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    
class CelebA(torch.utils.data.Dataset):
    def __init__(self, split, dir_path): # <<< FIX: Pass directory path as an argument
        self.dir = dir_path
        self.all_files = []
        self.labels = {}
        classes = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold", "Septoria"]

        for i, disease in enumerate(classes):
            disease_path = os.path.join(self.dir, disease)
            if os.path.isdir(disease_path):
                files = glob(os.path.join(disease_path, '*.jpg'))
                for file in files:
                    self.all_files.append(file)
                    self.labels[file] = i
            else:
                print(f"Warning: Directory not found: {disease_path}")

        random.shuffle(self.all_files)

        total_images = len(self.all_files)
        if total_images == 0:
            raise IOError(f"Error: No images found in the directory: {self.dir}")
        else:
            train_end = int(0.8 * total_images)
            valid_end = int(0.9 * total_images)

            if split == 'train':
                self.files = self.all_files[:train_end]
            elif split == 'valid':
                self.files = self.all_files[train_end:valid_end]
            elif split == 'test' or split == 'test_ref':
                self.files = self.all_files[valid_end:]
            else:
                raise ValueError(f"Invalid split: {split}")

        self.length = len(self.files)
        img_size = 128
        
        # <<< FIX 3: Corrected the order of transformations
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([img_size, img_size]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-15, 15)),
                SaltAndPepperNoise(amount=0.05),
                transforms.GaussianBlur(kernel_size=3),
                Occlusion(area_ratio=0.1),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
                transforms.ToTensor(),
                transforms.RandomApply([AddGaussianNoise(std=0.05)], p=0.5), # Use the new class
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __getitem__(self, index):
        filename = self.files[index]
        img = cv2.imread(filename)
        if img is None:
            print(f"Warning: Could not load image {filename}. Skipping.")
            return self.__getitem__((index + 1) % len(self.files)) # Return next item instead

        label = self.labels[filename]
        img = self.transform(img)
        return img, torch.LongTensor([label])

    def __len__(self):
        return self.length

def ImgForPlot(img):
    img = img.numpy().transpose((1, 2, 0)) # Use numpy and transpose correctly
    img = (127.5 * (img + 1)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


"""Hyper Parameter"""
nf = 64
nz = 64
nd = 5 # Number of classes
sdim = 64
lambda_gp = 10. # WGAN-GP usually uses 10
lambda_sty = 1.
lambda_ds = 1.
lambda_cyc = 1.
lr = 1e-4
lr_f = 1e-6
betas = (0.0, 0.99)
weight_decay = 1e-4
batch_size = 8
epochs = 50
ds_epochs = 5
n_print = 100
n_img_save = 2000
workers = 2 # Set to 0 if you still face issues

# <<< FIX 2: Use GPU if available, otherwise CPU
isCuda = torch.cuda.is_available()
device = torch.device("cuda:0" if isCuda else "cpu")

"""Model"""

def init_conv_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)

def init_fc_weight_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.)

def init_fc_weight_one(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 1.)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=False, down=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm = norm
        if norm:
            self.norm1 = nn.InstanceNorm2d(in_ch, affine=True)
            self.norm2 = nn.InstanceNorm2d(in_ch, affine=True)
        self.lrelu = nn.LeakyReLU(0.2)
        self.is_down = down
        self.is_sc = in_ch != out_ch
        if self.is_sc:
            self.sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def down(self, x):
        return F.avg_pool2d(x, 2)

    def forward(self, x):
        res = x
        if self.norm: res = self.norm1(res)
        res = self.lrelu(res)
        res = self.conv1(res)
        if self.is_down:
            res = self.down(res)
            x = self.down(x)
        if self.norm: res = self.norm2(res)
        res = self.lrelu(res)
        res = self.conv2(res)
        if self.is_sc: x = self.sc(x)
        return (x + res) / math.sqrt(2)

class AdaIN(nn.Module):
    def __init__(self, sdim, nf):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(nf)
        self.gamma = nn.Linear(sdim, nf)
        self.beta = nn.Linear(sdim, nf)
        self.apply(init_fc_weight_one)

    def forward(self, x, s):
        B, C, H, W = x.size()
        return (1 + self.gamma(s).view(B, C, 1, 1)) * self.norm(x) + self.beta(s).view(B, C, 1, 1)

class AdaINResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, sdim, up=False):
        super(AdaINResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = AdaIN(sdim, in_ch)
        self.norm2 = AdaIN(sdim, out_ch)
        self.lrelu = nn.LeakyReLU(0.2)
        self.is_up = up
        self.is_sc = in_ch != out_ch
        if self.is_sc:
            self.sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def up(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, x, s):
        res = self.lrelu(self.norm1(x, s))
        if self.is_up:
            res = self.up(res)
            x = self.up(x)
        res = self.conv1(res)
        res = self.lrelu(self.norm2(res, s))
        res = self.conv2(res)
        if self.is_sc: x = self.sc(x)
        return (x + res) / math.sqrt(2)

class Generator(nn.Module):
    def __init__(self, nf, sdim):
        super(Generator, self).__init__()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        self.enc = nn.Sequential(
            ResBlock(nf, 2*nf, norm=True, down=True),
            ResBlock(2*nf, 4*nf, norm=True, down=True),
            ResBlock(4*nf, 8*nf, norm=True, down=True),
            ResBlock(8*nf, 8*nf, norm=True),
            ResBlock(8*nf, 8*nf, norm=True)
        )
        self.dec = nn.ModuleList([
            AdaINResBlock(8*nf, 8*nf, sdim),
            AdaINResBlock(8*nf, 8*nf, sdim),
            AdaINResBlock(8*nf, 4*nf, sdim, up=True),
            AdaINResBlock(4*nf, 2*nf, sdim, up=True),
            AdaINResBlock(2*nf, nf, sdim, up=True)
        ])
        self.conv_out = nn.Sequential(
            nn.InstanceNorm2d(nf, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, 3, 1, 1, 0)
        )
        self.apply(init_conv_weight)

    def forward(self, x, s):
        x = self.conv_in(x)
        x = self.enc(x)
        for layer in self.dec:
            x = layer(x, s)
        x = self.conv_out(x)
        return x

class MappingNetwork(nn.Module):
    def __init__(self, nz, nd, sdim):
        super(MappingNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(nz, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.unshared = nn.ModuleList()
        for i in range(nd):
            self.unshared.append(nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, sdim)
            ))
        self.apply(init_fc_weight_zero)

    def forward(self, z, y):
        B = z.size(0)
        z = self.shared(z)
        s = [layer(z) for layer in self.unshared]
        s = torch.stack(s, dim=1)
        # <<< FIX 2: Send index tensor to the correct device
        i = torch.arange(B, device=z.device)
        return s[i, y]

class Discriminator(nn.Module):
    def __init__(self, nf, nd):
        super(Discriminator, self).__init__()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        self.res = nn.Sequential(
            ResBlock(nf, 2*nf, down=True),
            ResBlock(2*nf, 4*nf, down=True),
            ResBlock(4*nf, 8*nf, down=True),
            ResBlock(8*nf, 8*nf, down=True),
            ResBlock(8*nf, 8*nf, down=True)
        )
        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8*nf, 8*nf, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8*nf, nd, 1, 1, 0)
        )
        self.apply(init_conv_weight)

    def forward(self, x, y):
        B = x.size(0)
        x = self.conv_in(x)
        x = self.res(x)
        x = self.conv_out(x)
        x = x.view(B, -1)
        # <<< FIX 2: Send index tensor to the correct device
        i = torch.arange(B, device=x.device)
        return x[i, y]

class StyleEncoder(nn.Module):
    def __init__(self, nf, nd, sdim):
        super(StyleEncoder, self).__init__()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        self.res = nn.Sequential(
            ResBlock(nf, 2*nf, down=True),
            ResBlock(2*nf, 4*nf, down=True),
            ResBlock(4*nf, 8*nf, down=True),
            ResBlock(8*nf, 8*nf, down=True),
            ResBlock(8*nf, 8*nf, down=True)
        )
        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8*nf, 8*nf, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8*nf, nd*sdim, 1, 1, 0)
        )
        self.nd = nd
        self.apply(init_conv_weight)

    def forward(self, x, y):
        B = x.size(0)
        x = self.conv_in(x)
        x = self.res(x)
        x = self.conv_out(x)
        x = x.view(B, self.nd, -1)
        # <<< FIX 2: Send index tensor to the correct device
        i = torch.arange(B, device=x.device)
        return x[i, y]

class Model:
    def __init__(self, nf, nz, nd, sdim, lr, lr_f, betas, weight_decay, device):
        self.G = Generator(nf, sdim).to(device)
        self.F = MappingNetwork(nz, nd, sdim).to(device)
        self.D = Discriminator(nf, nd).to(device)
        self.E = StyleEncoder(nf, nd, sdim).to(device)
        self.optG = optim.Adam(self.G.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optF = optim.Adam(self.F.parameters(), lr=lr_f, betas=betas, weight_decay=weight_decay)
        self.optD = optim.Adam(self.D.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optE = optim.Adam(self.E.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def save(self, path):
        torch.save(self.G.state_dict(), path + '_G.pt')
        torch.save(self.F.state_dict(), path + '_F.pt')
        torch.save(self.D.state_dict(), path + '_D.pt')
        torch.save(self.E.state_dict(), path + '_E.pt')

    def load(self, path):
        self.G.load_state_dict(torch.load(path + '_G.pt'))
        self.F.load_state_dict(torch.load(path + '_F.pt'))
        self.D.load_state_dict(torch.load(path + '_D.pt'))
        self.E.load_state_dict(torch.load(path + '_E.pt'))

    def zero_grad(self):
        self.optG.zero_grad()
        self.optF.zero_grad()
        self.optD.zero_grad()
        self.optE.zero_grad()

    def train(self): self.G.train(); self.F.train(); self.D.train(); self.E.train()
    def eval(self): self.G.eval(); self.F.eval(); self.D.eval(); self.E.eval()

def gradient_penalty(out, x):
    grad = torch.autograd.grad(outputs=out.sum(), inputs=x, create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.pow(2).view(x.size(0), -1).sum(1).mean()
    return grad

# <<< FIX: This is the main guard for multiprocessing
if __name__ == '__main__':
    # Set up paths
    dataset_dir = 'D:/ArkoStudyMaterials/CVPR/Model/CVPR_Dataset' # Update with your path
    model_saved_dir = "."
    model_name = 'stargan_tomato_v1'
    loss_path = f"{model_saved_dir}/{model_name}_loss.pkl"
    img_path = f"{model_saved_dir}/{model_name}_img.pkl"
    model_path = f"{model_saved_dir}/{model_name}"
    
    # Setup datasets and dataloaders
    dataset = CelebA('train', dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    ref_dataset = CelebA('train', dataset_dir)
    ref_dataloader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    ref2_dataset = CelebA('train', dataset_dir)
    ref2_dataloader = DataLoader(ref2_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    test_dataset = CelebA('test', dataset_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=workers)
    test_ref_dataset = CelebA('test_ref', dataset_dir)
    ref_test_dataloader = DataLoader(test_ref_dataset, batch_size=16, shuffle=False, num_workers=workers)
    
    print("Device: {}".format(device))

    # Initialize model
    model = Model(nf, nz, nd, sdim, lr, lr_f, betas, weight_decay, device)

    start_epoch = 0
    losses = {'train_d1': [], 'train_d2': [], 'train_g1': [], 'train_g2': []}
    if os.path.exists(loss_path):
        with open(loss_path, 'rb') as f:
            losses = pickle.load(f)
    
    l1 = nn.L1Loss()
    lambda_ds_zero = lambda_ds
    lambda_ds = max((lambda_ds_zero * (ds_epochs - start_epoch)) / epochs, 0.)

    step, d1_sum, d2_sum, g1_sum, g2_sum = 0, 0., 0., 0., 0.
    imgs = []
    
    # Training Loop
    for ep in range(start_epoch, epochs):
        model.train()
        timestamp = time.time()
        for batch_idx, ((x, y), (x_ref, y_), (x_ref2, _)) in enumerate(zip(dataloader, ref_dataloader, ref2_dataloader)):
            step += 1
            x, y = x.to(device), y.to(device).squeeze(1)
            x.requires_grad_()
            x_ref, y_ = x_ref.to(device), y_.to(device).squeeze(1)
            x_ref2 = x_ref2.to(device)
            z = torch.randn((x.size(0), nz)).to(device)

            # --- Train Discriminator ---
            # D loss with z
            model.zero_grad()
            out_real = model.D(x, y)
            loss_real = -out_real.mean()
            loss_gp = gradient_penalty(out_real, x)

            with torch.no_grad():
                s_ = model.F(z, y_)
                x_ = model.G(x, s_)
            out_fake = model.D(x_, y_)
            loss_fake = out_fake.mean()

            loss_d1 = loss_real + loss_fake + lambda_gp * loss_gp
            loss_d1.backward()
            model.optD.step()
            losses['train_d1'].append(loss_d1.item())
            d1_sum += loss_d1.item()
            
            # D loss with x_ref
            model.zero_grad()
            out_real = model.D(x, y)
            loss_real = -out_real.mean()
            loss_gp = gradient_penalty(out_real, x)

            with torch.no_grad():
                s_ = model.E(x_ref, y_)
                x_ = model.G(x, s_)
            out_fake = model.D(x_, y_)
            loss_fake = out_fake.mean()
            
            loss_d2 = loss_real + loss_fake + lambda_gp * loss_gp
            loss_d2.backward()
            model.optD.step()
            losses['train_d2'].append(loss_d2.item())
            d2_sum += loss_d2.item()


            # --- Train Generator ---
            if (batch_idx + 1) % 5 == 0: # Train G every 5 D steps
                # G loss with z, z2
                model.zero_grad()
                s_ = model.F(z, y_)
                x_ = model.G(x, s_)
                out = model.D(x_, y_)
                loss_adv = -out.mean()
                s_pred = model.E(x_, y_)
                loss_sty = l1(s_, s_pred)
                
                z2 = torch.randn((x.size(0), nz)).to(device)
                s_2 = model.F(z2, y_)
                x_2 = model.G(x, s_2)
                loss_ds = l1(x_, x_2.detach())

                s = model.E(x, y)
                x_rec = model.G(x_, s)
                loss_cyc = l1(x, x_rec)
                
                loss_g1 = loss_adv + lambda_sty * loss_sty - lambda_ds * loss_ds + lambda_cyc * loss_cyc
                loss_g1.backward()
                model.optG.step(); model.optF.step(); model.optE.step()
                losses['train_g1'].append(loss_g1.item())
                g1_sum += loss_g1.item()

                # G loss with x_ref, x_ref2
                model.zero_grad()
                s_ = model.E(x_ref, y_)
                x_ = model.G(x, s_)
                out = model.D(x_, y_)
                loss_adv = -out.mean()

                s_pred = model.E(x_, y_)
                loss_sty = l1(s_, s_pred)

                s_2 = model.E(x_ref2, y_)
                x_2 = model.G(x, s_2)
                loss_ds = l1(x_, x_2.detach())
                
                s = model.E(x, y)
                x_rec = model.G(x_, s)
                loss_cyc = l1(x, x_rec)
                
                loss_g2 = loss_adv + lambda_sty * loss_sty - lambda_ds * loss_ds + lambda_cyc * loss_cyc
                loss_g2.backward()
                model.optG.step(); model.optE.step()
                losses['train_g2'].append(loss_g2.item())
                g2_sum += loss_g2.item()

            lambda_ds = max(lambda_ds - lambda_ds_zero / (len(dataloader) * epochs), 0.)

            if (batch_idx + 1) % n_print == 0 or (batch_idx + 1) == len(dataloader):
                elapsed = time.time() - timestamp
                g_step = step / 5
                print(f'[{ep+1}/{epochs}][{batch_idx+1:5d}/{len(dataloader):5d}] Train ({elapsed:.1f}s)')
                print(f'\t[Loss] D1: {d1_sum/step:.4f} / D2: {d2_sum/step:.4f} / G1: {g1_sum/g_step:.4f} / G2: {g2_sum/g_step:.4f}')
                step, d1_sum, d2_sum, g1_sum, g2_sum = 0, 0., 0., 0., 0.
                timestamp = time.time()
                
        # --- End of Epoch ---
        model.eval()
        with torch.no_grad():
            temp_img = []
            for (x, _), (x_ref, y_) in zip(test_dataloader, ref_test_dataloader):
                x, x_ref, y_ = x.to(device), x_ref.to(device), y_.to(device).squeeze(1)
                s_ = model.E(x_ref, y_)
                x_ = model.G(x, s_)
                temp_img.append(x_.cpu())
            imgs.append(torch.cat(temp_img, dim=0))
        print(f"Epoch {ep+1}: Generated images for test dataset")

        model.save(model_path)
        with open(loss_path, 'wb') as f:
            pickle.dump(losses, f)
        with open(img_path, 'wb') as f:
            pickle.dump(imgs, f)
        print("Saved model, losses, and images completely!")

