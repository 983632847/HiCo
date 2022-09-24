import torch.nn as nn
import torchvision.models as models
from fpn import *

class ResNetUSCL(nn.Module):
    ''' The ResNet feature extractor + projection head + classifier for USCL '''

    def __init__(self, base_model, out_dim, pretrained=False):
        super(ResNetUSCL, self).__init__()

        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
        #                     "resnet50": models.resnet50(pretrained=pretrained)}
        # if pretrained:
        #     print('\nModel parameters loaded.\n')
        # else:
        #     print('\nRandom initialize model parameters.\n')
        
        # resnet = self._get_basemodel(base_model)
        # num_ftrs = resnet.fc.in_features

        # self.features = nn.Sequential(*list(resnet.children())[:-1])  # discard the last fc layer
        self.features = FPNresnet18(True)       # default=True
        # self.features = FPNresnet18(False)      # scratch
        self.l2 = nn.Linear(256*56*56, out_dim)
        self.l3 = nn.Linear(256*28*28, out_dim)
        self.l4 = nn.Linear(256*14*14, out_dim)
        self.l5 = nn.Linear(256*7*7, out_dim)
        # resnet50fpn 更改 512 为 2048
        self.lf = nn.Linear(512*1*1, out_dim)



        # projection MLP
        # self.linear = nn.Linear(1*out_dim, out_dim)
        self.linear = nn.Linear(5*out_dim, out_dim)

        # classifier
        num_classes = 3
        self.fc = nn.Linear(out_dim, num_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        p2, p3, p4, p5, h = self.features(x)
        f2, f3, f4, f5, fp = \
            torch.flatten(p2, start_dim=1), torch.flatten(p3, start_dim=1), torch.flatten(p4, start_dim=1), torch.flatten(p5, start_dim=1), torch.flatten(h, start_dim=1)
        x2 = F.relu(self.l2(f2))
        x3 = F.relu(self.l3(f3))
        x4 = F.relu(self.l4(f4))
        x5 = F.relu(self.l5(f5))
        xf = F.relu(self.lf(fp))

        # x = xf
        # x = self.linear(x)
        x = torch.cat((x2,x3,x4,x5,xf), 1)
        x = self.linear(x)

        return x
