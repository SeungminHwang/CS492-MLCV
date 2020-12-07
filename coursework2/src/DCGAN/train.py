import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import model
import torch.utils.data.dataloader as dataloader
import torchvision.datasets as dsets
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import numpy as np
import math
from time import time
torch.set_printoptions(threshold=math.inf)


def main():
    transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
    batch_size = 128
    epochs = 60

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    print(device)

    train_data = dsets.MNIST(root='../data/', train=True, transform=transform, download=True)
    train_loader = dataloader.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

    generator = model.Generator().to(device)
    discriminator = model.Discriminator().to(device)
    print(generator)
    print(discriminator)

    optimD = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimG = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, 100, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    start = time()

    outf = "save"
    if not os.path.isdir(outf):
        os.mkdir(outf)

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        for index, trainData in enumerate(train_loader, 0):
            # for i in range(k):
            discriminator.zero_grad()
            batch_size = trainData[0].size(0)
            target_real = torch.full((batch_size, 1), real_label).to(device)
            target_fake = torch.full((batch_size, 1), fake_label).to(device)
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            gen_im = generator(noise)
            real_out = discriminator(trainData[0].to(device))
            disLoss1 = criterion(real_out, target_real)
            disLoss1.backward()
            D_x = real_out.mean().item()
            gen_out = discriminator(gen_im)
            D_G_z1 = gen_out.mean().item()

            disLoss2 = criterion(gen_out, target_fake)
            disLoss2.backward()
            disLoss = disLoss1 + disLoss2
            optimD.step()

            generator.zero_grad()
            target_real = torch.full((batch_size, 1), real_label).to(device)
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            gen_im = generator(noise)
            gen_out = discriminator(gen_im)
            D_G_z2 = gen_out.mean().item()
            genLoss = criterion(gen_out, target_real)
            genLoss.backward()
            optimG.step()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epochs, index, len(train_loader),
                     disLoss.item(), genLoss.item(), D_x, D_G_z1, D_G_z2))
            if index % 100 == 0:
                vutils.save_image(trainData[0].to(device),
                                  '%s/real_samples.png' % outf,
                                  normalize=True)
                fake = generator(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                                  normalize=True)

            # do checkpointing
        torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
    print("end, time: {}".format(time() - start))
    z = torch.randn(1, 100, 1, 1, device=device)
    gen_im = generator(z)
    return gen_im, generator, discriminator


if __name__ == "__main__":
    im, gen, dis = main()
    im = im / 2 + 0.5
    im = (im.round()*255).int()
    cv2.imwrite("test.jpg", im.squeeze(0).squeeze(0).cpu().detach().numpy())
