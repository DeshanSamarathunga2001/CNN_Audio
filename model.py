import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride =1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,stride,padding=1,bias=False)
        self.bn1 = nn.BatchNormal2nd(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False)
        self.bn2 = nn.BatchNormal2nd(out_channels)
        self.shortcut = nn.Sequential()

        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, fmap_dict=None,prefix=""):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.con"] = out_add
        
        out = torch.relu(out_add)
        if fmap_dict is not None:
            fmap_dictp[f"{prefix}".relu] = out

        return out

class AudioCNN(nn.Module):
    def __init__(self, num_class=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNom2d(63),
            nn.ReLu(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = nn.ModuleList([ResidualBlock(64,64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [
                ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1)
                for i in range (4)
            ])
        self.layer3 = nn.ModuleList(
            [
                ResidualBlock(128 if i == 0 else 256,256, stride=2 if i == 0 else 1)
                for i in range(6)
            ]
        )
        

            