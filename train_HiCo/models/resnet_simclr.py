import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .model_resnet import resnet18_cbam, resnet50_cbam
from .fpn import *

class ResNetSimCLR(nn.Module):
    ''' The ResNet feature extractor + projection head for SimCLR '''

    def __init__(self, base_model, out_dim, pretrained=True):
        super(ResNetSimCLR, self).__init__()

        # use_CBAM = False  # # use CBAM or not, att_type="CBAM" or None
        # if use_CBAM:
        #     self.resnet_dict = {"resnet18": resnet18_cbam(pretrained=pretrained),
        #                         "resnet50": resnet50_cbam(pretrained=pretrained)}
        # else:
        #     self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
        #                         "resnet50": models.resnet50(pretrained=pretrained)}


        # if pretrained:
        #     print('\nImageNet pretrained parameters loaded.\n')
        # else:
        #     print('\nRandom initialize model parameters.\n')
        
        # resnet = self._get_basemodel(base_model)
        # num_ftrs = resnet.fc.in_features

        # self.features = nn.Sequential(*list(resnet.children())[:-1]) # discard the last fc layer

        # projection MLP
        self.features = FPNresnet18(pretrained)
        # self.features = FPNresnet18(False)         # default = True
        self.l2 = nn.Linear(256*56*56, out_dim)
        self.l3 = nn.Linear(256*28*28, out_dim)
        self.l4 = nn.Linear(256*14*14, out_dim)
        self.l5 = nn.Linear(256*7*7, out_dim)

        # resnet50fpn 更改 512 为 2048
        self.lf = nn.Linear(512*1*1, out_dim)  # out_dim = 256

        # self.l11 = nn.Linear(512*1*1, 512*1*1)
        # self.l22 = nn.Linear(512*1*1, out_dim)
        
        self.l22 = nn.Linear(out_dim, out_dim)
        self.l33 = nn.Linear(out_dim, out_dim)
        self.l55 = nn.Linear(out_dim, out_dim)

        #########################################################
        # num_classes = 2   # butte
        num_classes = 12  # US-4
        self.fc = nn.Linear(1*out_dim, num_classes)
        # self.fc = nn.Linear(5*out_dim, num_classes)

        ## Mixup
        # self.fc1 = nn.Linear(num_ftrs, num_ftrs)
        # self.fc2 = nn.Linear(num_ftrs, num_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    # def forward(self, x):
    #     h = self.features(x)
    #
    #     h = h.squeeze()
    #
    #     x = self.l1(h)
    #     x = F.relu(x)
    #     x = self.l2(x)
    #     return h, x # the feature vector, the output

    def forward(self, x):
        p2, p3, p4, p5, h = self.features(x)
        h1 = h.squeeze()  # feature before project g()=h1
        # x = self.l11(h1)
        # x = F.relu(x)
        # x = self.l22(x)

        f2, f3, f4, f5, fp = \
            torch.flatten(p2, start_dim=1), torch.flatten(p3, start_dim=1), torch.flatten(p4, start_dim=1), torch.flatten(p5, start_dim=1), torch.flatten(h, start_dim=1)
        x2 = F.relu(self.l2(f2))
        x3 = F.relu(self.l3(f3))
        x4 = F.relu(self.l4(f4))
        x5 = F.relu(self.l5(f5))
        xf = F.relu(self.lf(fp))

        # Version 1
        # size : 1 * out_dim
        c = xf
        c = self.fc(c)

        # # Version 2
        # # size : 5 * out_dim
        # x = torch.cat((x2,x3,x4,x5,xf), 1)
        # c = x
        # c = c.view(c.size(0), -1)
        # c = self.fc(c)

        xx2 = F.relu(self.l2(f2))    # From p2: l2 -> relu -> l22
        xx2 = self.l22(xx2)

        xx3 = F.relu(self.l3(f3))    # From p3: l3 -> relu -> l33
        xx3 = self.l33(xx3)

        xx5 = F.relu(self.l5(f5))    # From p5: l5 -> relu -> l55
        xx5 = self.l55(xx5)

        return xx2, xx3, xx5, c