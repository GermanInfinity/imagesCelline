import torch
import torch.nn as nn
import random

#Devise a way whatever goes into res net is also going out 
#Devise a way to keep feature mapping consistent
def gene_creator(input_features):
    gene = []
    blocks = ["Res", "Invr", "Bot", "CrLu"]
    out_channels = [8,16]

    for pos in range(7):
        block = random.choice(blocks)
        out = random.choice(out_channels)
        if block == "Invr":
            stride = random.choice([1,2])
            if pos == 0:
                #Specific setting for if first block is Invr
                gene.append([block, input_features, out, stride, 1])
                continue
            #Input to Invr that is not the first has a stride of 6
            gene.append([block, gene[-1][2], out, stride, 6])
            continue

        if pos == 0:
            #First input block for all types
            gene.append([block, input_features, out])

        else: 
            #Continuous flow of blocks
            gene.append([block, gene[-1][2], out])


    return gene

def gene_corrector(gene):
    # Correct the input channels to blocks after bottleneck blocks
    # for pos in range(1,len(gene)):
    #     if gene[pos-1][0] == 'Bot':
    #         gene[pos][1] = gene[pos-1][2] * 4

    #Correct the flow of output channels in network=2x per layer
    for pos in range(len(gene)):
        if pos == 0:
            first_out = gene[pos][2]
            continue
        if gene[pos][0] == "Res": continue
        if gene[pos][0] == "Bot": continue
        gene[pos][2] = first_out * 2 * pos

    #Correct input and output for ResNet
    for pos in range(1,len(gene)):
        prev_block_output = gene[pos-1][2]
        block = gene[pos]

        if block[0] == 'Res' or block[0] == 'Bot':
            block[1] = prev_block_output
            block[2] = block[1]

    #Refix input output feature mappings
    for pos in range(1,len(gene)):
        block = gene[pos]
        prev_block = gene[pos-1]
        if block[0] == 'Res': continue
        if block[0] == 'Bot': continue
        if block[1] != prev_block[2]: 
            block[1] = prev_block[2]
    return gene

def make_chromosome(input_features):

    torque_list  =  {"BR":0.0, "BI":0.0, "BB":0.0, "BC":0.0,
                 "RR":0.0, "RI":0.0, "RB":0.0, "RC":0.0,
                 "CR":0.0, "CI":0.0, "CB":0.0, "CC":0.0,
                 "IR":0.0, "II":0.0, "IB":0.0, "IC":0.0,
                 }
    presence_ratio  =  {"Bot":0.0, "Res":0.0, "Invr":0.0, "CrLu":0.0}

    chromosome = {"Gene":gene_corrector(gene_creator(input_features)), "Torque_list":torque_list, 
                  "Presence_ratios":presence_ratio, "Fitness":0.0}
    return (chromosome)


# Conv 1x1
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
# Conv 3x3
def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)   
# ConvBNReLU
class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        # super(ConvBNReLU, self).__init__(
        #     nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
        #     nn.BatchNorm2d(out_planes),
        #     nn.ReLU6(inplace=True)
        # )
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, 
                               padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Inverted Residual block
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
            return x + self.conv(x)
        else:
            out = self.conv(x)
            return out

# Bottleneck block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


#Net
class Net(nn.Module):
    def __init__(self, gene, num_classes=5):
        super(Net, self).__init__()

        self.in_channels = 3
        self.conv_out = gene[0][2]

        self.conv = nn.Conv2d(self.in_channels, self.conv_out, 3)
        self.bn = nn.BatchNorm2d(self.conv_out)
        self.relu = nn.ReLU(inplace=True)

        self.layers = self.make_layer(gene)

        self.avg_pool = nn.AvgPool2d(6)

        #self.fc = nn.Linear(num_classes)
        
    def make_layer(self, gene):

        layers = []
        first_gene = True
        for ele in gene:
            if ele[0] == "Res":
                if first_gene:
                    layers.append(ResidualBlock(self.conv_out, ele[2], stride=1, downsample=None))
                    first_gene = False
                else:
                    layers.append(ResidualBlock(ele[2], ele[2], stride=1, downsample=None))

            if ele[0] == "Invr":
                if first_gene: 
                    layers.append(InvertedResidual(self.conv_out, ele[2], stride=ele[-1], expand_ratio=ele[-2]))
                    first_gene = False
                else:
                    layers.append(InvertedResidual(ele[1], ele[2], stride=ele[-1], expand_ratio=ele[-2]))

            if ele[0] == "Bot": 
                if first_gene: 
                    layers.append(Bottleneck(self.conv_out, ele[2], stride=1))
                    first_gene = False
                else:              
                    layers.append(Bottleneck(ele[1], ele[2], stride=1))

            if ele[0] == "CrLu":     
                if first_gene: 
                    layers.append(ConvBNReLU(self.conv_out, ele[2]))
                    first_gene = False
                else:        
                    layers.append(ConvBNReLU(ele[1], ele[2]))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        last_feats = out.size()[1]
        print (last_feats)
        out = nn.Linear(out,5)#self.fc(self.in_feats, out)
        
        return out

#chromosome = make_chromosome(3)
#gene = chromosome["Gene"]
Net = Net(gene)
print (gene)
print (Net)
