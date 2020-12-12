import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ch = 256
        self.conv1  = nn.ConvTranspose2d(100, ch*8, 4, 1, 0, bias=False)
        self.bn1    = nn.BatchNorm2d(ch*8)
        self.conv2  = nn.ConvTranspose2d(ch*8, ch*4, 4, 2, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(ch*4)
        self.conv3  = nn.ConvTranspose2d(ch*4,  ch*2, 4, 2, 1, bias=False)
        self.bn3    = nn.BatchNorm2d(ch*2)
        self.conv4  = nn.ConvTranspose2d(ch*2,  ch, 4, 2, 1, bias=False)
        self.bn4    = nn.BatchNorm2d(ch)
        self.conv5  = nn.ConvTranspose2d(ch,    1, 4, 2, 1, bias=False)
        self.tanh   = nn.Tanh()

        self.relu = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, noise):
        c1 = self.relu(self.bn1(self.conv1(noise)))
        c2 = self.relu(self.bn2(self.conv2(c1)))
        c3 = self.relu(self.bn3(self.conv3(c2)))
        c4 = self.relu(self.bn4(self.conv4(c3)))
        out = self.tanh(self.conv5(c4))

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ch = 128
        self.conv1 = nn.Conv2d(1,    ch, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ch,  ch*2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch*2)
        self.conv3 = nn.Conv2d(ch*2,  ch*4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch*4)
        self.conv4 = nn.Conv2d(ch*4, ch*8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ch*8)
        self.conv5 = nn.Conv2d(ch*8,   1, 4, 1, 0, bias=False)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, im):
        c1 = self.lrelu(self.conv1(im))
        c2 = self.lrelu(self.bn2(self.conv2(c1)))
        c3 = self.lrelu(self.bn3(self.conv3(c2)))
        c4 = self.lrelu(self.bn4(self.conv4(c3)))
        out = self.sig(self.conv5(c4))

        return out.reshape((-1, 1))
