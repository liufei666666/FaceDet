import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.timer import Timer
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
#log_dir = '../visual_featuremaps'
#writer = SummaryWriter(log_dir=log_dir)
_a = {'conv1': Timer(), 'conv2': Timer(), 'ince': Timer(), 'conv3': Timer(), 'conv4': Timer(), 'pre': Timer()}


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class conv_dw(nn.Module):

    def __init__(self, in_channels, out_channels,stride, **kwargs):
        super(conv_dw, self).__init__()
        self.conv3_3 = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.bn3_3= nn.BatchNorm2d(in_channels)
        self.re3_3 = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.re1_1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.re3_3(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.re1_1(x)
        return x

class conv_1_1(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_1_1, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.re1_1 = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.re1_1(x)
        return x



class CRelu(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #x = torch.cat([x, -x], 1)
        x = F.relu(x, inplace=True)
        return x


class FaceBoxes(nn.Module):

    def __init__(self, phase, size, num_classes):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        print(num_classes)
        self.size = size

        self.conv1 = CRelu(1, 24, kernel_size=7, stride=4, padding=3)
        self.conv2 = CRelu(24, 48, kernel_size=5, stride=2, padding=2)

        self.conv2_1 = conv_dw(48, 64,1)
        self.conv2_2 = conv_dw(64, 128, 1)
        self.conv2_3 = conv_dw(128, 256, 1)


        self.conv3_1 = conv_dw(256, 256, 1)
        self.conv3_2 = conv_dw(256, 512, 2)

        self.conv4_1 = conv_dw(512, 256, 1)
        self.conv4_2 = conv_dw(256, 512, 2)

        self.loc, self.conf = self.multibox(self.num_classes)

        if self.phase == 'test':
            # print('????????')
            self.softmax = nn.Softmax()

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
    
    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []

        loc_layers += [nn.Conv2d(256, 21 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 21 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def forward(self, x):

        sources = list()
        loc = list()
        conf = list()
        detection_dimension = list()

        _a['conv1'].tic()
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        _a['conv1'].toc()

        _a['conv2'].tic()
        x = self.conv2(x)
        #x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        _a['conv2'].toc()

        _a['ince'].tic()

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        #x = self.conv2_4(x)
        #print(x.shape)

        #x1 = x.transpose(0, 1)  # C£¬B, H, W  ---> B£¬C, H, W
        #img_grid1 = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=2)  # B£¬C, H, W
        #writer.add_image( '1_feature_maps', img_grid1, global_step=666)

        _a['ince'].toc()

        detection_dimension.append(x.shape[2:])
        sources.append(x)
        _a['conv3'].tic()
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        #print(x.shape)
        _a['conv3'].toc()
        detection_dimension.append(x.shape[2:])
        sources.append(x)

        _a['conv4'].tic()
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        _a['conv4'].toc()
        #print(x.shape)

        detection_dimension.append(x.shape[2:])
        sources.append(x)
        _a['pre'].tic()
        detection_dimension = torch.tensor(detection_dimension, device=x.device)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)

        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        _a['pre'].toc()
        if self.phase == "test":
            output = (loc.view(loc.size(0), -1, 4),
                      self.softmax(conf.view(-1, self.num_classes)),
                      detection_dimension)
            # print('conv1',_a['conv1'].average_time,'conv2',_a['conv2'].average_time,'ince',_a['ince'].average_time,'conv3',_a['conv3'].average_time,'conv4',_a['conv4'].average_time,'pre',_a['pre'].average_time)
        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes),
                      detection_dimension)
            # print('conv1',_a['conv1'].average_time,'conv2',_a['conv2'].average_time,'ince',_a['ince'].average_time,'conv34',_a['conv34'].average_time)
        return output
#writer.close()