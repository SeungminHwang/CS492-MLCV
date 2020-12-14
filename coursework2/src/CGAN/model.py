import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        ngf = args.ngf
        nz = args.nz
        #cgan
        self.deconv1_label = nn.ConvTranspose2d(10, ngf * 2, 4, 1, 0, bias=False)
        self.deconv1_noise = nn.ConvTranspose2d(nz, ngf * 2, 4, 1, 0, bias=False)
        self.bn1_label = nn.BatchNorm2d(ngf * 2)
        self.bn1_noise = nn.BatchNorm2d(ngf * 2)
        
        
        self.deconv2  = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(ngf * 2)
        self.deconv3  = nn.ConvTranspose2d(ngf * 2,  ngf, 4, 2, 1, bias=False)
        self.bn3    = nn.BatchNorm2d(ngf)
        self.deconv4  = nn.ConvTranspose2d(ngf,    1, 4, 2, 1, bias=False)
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
        out = self.tanh(self.deconv4(c3))
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        ndf = args.ndf
        #cgan
        self.conv1_img = nn.Conv2d(1, ndf, 4, 2, 1, bias=False)
        self.conv1_label = nn.Conv2d(10, ndf, 4, 2, 1, bias=False)
        
        #self.conv1 = nn.Conv2d(1,    128, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf * 2,  ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv3 = nn.Conv2d(ndf * 4,  ndf * 8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)
        self.conv4 = nn.Conv2d(ndf * 8,   1, 4, 1, 0, bias=False)

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
        out = self.sig(self.conv4(c3))
        return out.reshape((-1, 1))