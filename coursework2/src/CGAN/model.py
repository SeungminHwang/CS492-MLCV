import torch
import torch.nn as nn
import torchvision.transforms as transforms



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        #cgan
        self.deconv1_label = nn.ConvTranspose2d(10, 512, 4, 1, 0, bias=False)
        self.deconv1_noise = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False)
        self.bn1_label = nn.BatchNorm2d(512)
        self.bn1_noise = nn.BatchNorm2d(512)
        
        
        self.deconv2  = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(512)
        self.deconv3  = nn.ConvTranspose2d(512,  256, 4, 2, 1, bias=False)
        self.bn3    = nn.BatchNorm2d(256)
        self.deconv4  = nn.ConvTranspose2d(256,  128, 4, 2, 1, bias=False)
        self.bn4    = nn.BatchNorm2d(128)
        self.deconv5  = nn.ConvTranspose2d(128,    1, 4, 2, 1, bias=False)
        self.tanh   = nn.Tanh()

        self.relu = nn.ReLU(True)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    
    def forward(self, noise, labels):
        # cgan
        x = self.relu(self.bn1_noise(self.deconv1_noise(noise)))
        l = self.relu(self.bn1_label(self.deconv1_label(labels)))
        input = torch.cat([x, l], 1)
        
        c2 = self.relu(self.bn2(self.deconv2(input)))
        c3 = self.relu(self.bn3(self.deconv3(c2)))
        c4 = self.relu(self.bn4(self.deconv4(c3)))
        out = self.tanh(self.deconv5(c4))
        
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #cgan
        self.conv1_img = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.conv1_label = nn.Conv2d(10, 64, 4, 2, 1, bias=False)
        
        #self.conv1 = nn.Conv2d(1,    128, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(128,  256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256,  512, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024,   1, 4, 1, 0, bias=False)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, im, labels):
        # cgan
        x = self.lrelu(self.conv1_img(im))
        l = self.lrelu(self.conv1_label(labels))
        input = torch.cat([x, l], 1)
        
        c2 = self.lrelu(self.bn2(self.conv2(input)))
        c3 = self.lrelu(self.bn3(self.conv3(c2)))
        c4 = self.lrelu(self.bn4(self.conv4(c3)))
        out = self.sig(self.conv5(c4))

        return out.reshape((-1, 1))