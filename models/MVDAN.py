import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
from torchsummary import summary
from models.scnet import scnet50,scnet50_v1d,scnet101
from models.senet import se_resnet50,se_resnet101
from models.cbam import resnet18_cbam,resnet50_cbam,resnet101_cbam

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)

        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_densenet = cnn_name.startswith('densenet')
        self.mean = Variable(torch.FloatTensor([0.0142, 0.0142, 0.0142]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.0818, 0.0818, 0.0818]), requires_grad=False).cuda()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet50-sc':
                self.net = scnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet50-se':
                 self.net = se_resnet50(pretrained=self.pretraining)
                 self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet50-cbam':
                 self.net = resnet50_cbam(pretrained=self.pretraining)
                 self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet50x':
                self.net = models.resnext50_32x4d(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet50x-sc':
                self.net = scnet50_v1d(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet101x':
                self.net = models.resnext101_32x8d(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnest':
                self.net = models.resnest18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet101':
                self.net = scnet101(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)

        elif self.use_densenet:
            if self.cnn_name =='densenet121':
                self.net = models.densenet121(pretrained=self.pretraining)
                self.net.classifier = nn.Linear(1024, 40)

        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, 40)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        elif self.use_densenet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))


class MVCNN(Model):
    def __init__(self, name, model, pool_mode='max', nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.nclasses = nclasses
        self.num_views = num_views
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_densenet = cnn_name.startswith('densenet')
        self.num_conv = 2
        self.pool_mode = pool_mode

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])

            # self.net_3 = nn.Sequential(*list(model.net.children())[:4])
        elif self.use_densenet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.pool = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
            self.net_2 = nn.Sequential(nn.Linear(1024, 40))
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

        if self.pool_mode == 'DAN':
            self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(1, 1, 1, bias=False)
            self.conv3 = nn.Conv2d(1, 1, 1, bias=False)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(num_views, num_views // num_views, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(num_views // num_views, num_views, bias=False),
                nn.Sigmoid()
            )
            # self.fc = nn.Sequential(
            #     nn.Conv2d(num_views, num_views // 12, 1, bias=False), nn.ReLU(),
            #     nn.Conv2d(num_views // 12, num_views, 1, bias=False),
            # )
            self.conv = nn.Conv2d(1, 1, (1, num_views), padding_mode='valid')
            self.net_2 = nn.Sequential(nn.Linear(2048*2, 40))  





    def forward(self, x):
        y = self.net_1(x)


        if  self.pool_mode == 'DAN':
            yp = y.view((int(x.shape[0] / self.num_views), 1, (y.shape[-3] * y.shape[-2] * y.shape[-1]), self.num_views))
            yc = y.view((int(x.shape[0] / self.num_views), self.num_views, (y.shape[-3] * y.shape[-2] * y.shape[-1]), 1))

            # VSAB
            q = self.conv1(yp)
            k = self.conv2(yp)
            v = self.conv3(yp)
            s = torch.matmul(torch.transpose(q, 2, 3), k)
            beta = torch.nn.functional.softmax(s,dim=-1)
            o = torch.matmul(v, beta)
            gamma = torch.autograd.Variable(torch.FloatTensor([[1.]]), requires_grad=True).cuda()
            yp = yp + gamma * o
            yp = self.conv(yp)
            yp = yp.view(yp.shape[0], -1)

            # VCAB
            b, c, _, _ = yc.size()
            y1 = self.avg_pool(yc).view(b, c)
            y1 = self.fc(y1).view(b, c, 1, 1)
            yc = torch.matmul(yc, y1)
            yc = yc.view((int(x.shape[0] / self.num_views), 1, yc.shape[-2], self.num_views))
            yc = self.conv(yc)
            yc = yc.view(yc.shape[0], -1)
            y = torch.cat((yp, yc), 1)
        return self.net_2(y)



if __name__ == '__main__':
    cnn_name = 'resnet50'
    name = 'MVCNN'
    cnet = SVCNN(name, nclasses=40, pretraining=True, cnn_name=cnn_name)
    model = MVCNN(name, cnet, nclasses=40, cnn_name=cnn_name, num_views=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=(3, 224, 224))
