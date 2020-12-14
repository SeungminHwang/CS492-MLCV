import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import model
import pickle
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

device = torch.device("cuda:2")

PATH_G = "/tmp/pycharm_project_648/coursework2/src/CGAN/gen64/netG_epoch_39.pth"  # modify
args.ngf = 64
args.nz = 100
generator = model.Generator(args).to(device)
checkpoint_G = torch.load(PATH_G, map_location='cpu')
generator.load_state_dict(checkpoint_G)
C = 1
H = 32
W = 32
N_Class = 10
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1).to(device)
labels = []

for i in range(10):
    for j in range(2000):
        noise = torch.randn((100,), device=device).float()
        noise = noise.reshape((1, 100, 1, 1))
        y = onehot[i].unsqueeze(0)
        gen_img = generator(noise, y.detach()).view(-1, C, H, W)
        labels.append(i)
        save_image(gen_img.data, "./gen_imgs/%d.png" % (2000*i+j), normalize=True)
labels = np.array(labels).reshape((20000, 1))
np.save("./labels", labels)


