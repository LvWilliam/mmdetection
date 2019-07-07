import torch

def atrous_block1(in_channel, out_channel):
    output = torch.nn.Conv2d(in_channel, out_channel, 1, 1)
    return output


def atrous_block6(in_channel, out_channel):
    output = torch.nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)

    return output


def atrous_block12(in_channel, out_channel):
    output = torch.nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
    return output


def atrous_block18(in_channel, out_channel):
    output = torch.nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
    return output


def conv_1x1_output(in_channel, out_channel):
    output = torch.nn.Conv2d(in_channel * 5, out_channel, 1, 1)
    return output


def global_block(in_channel, out_channel, size):
    output = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)),
                                 torch.nn.Conv2d(in_channel, out_channel, 1, 1),
                                 torch.nn.Upsample((size, size), mode='bilinear'))
    return output


class aspp_module(nn.Module):
    def __init__(self, in_channel, out_channel, size):
        super(ASPP, self).__init__()
        # for i, info in enumetate(zip(in_channel, out_channel, size)):
        self.atrous_block1 = atrous_block1(in_channel, out_channel)
        self.atrous_block6 = atrous_block6(in_channel, out_channel)
        self.atrous_block12 = atrous_block12(in_channel, out_channel)
        self.atrous_block18 = atrous_block12(in_channel, out_channel)
        self.conv_1x1_output = conv_1x1_output(in_channel, out_channel)
        self.global_block= global_block(in_channel, out_channel, size)

    def forward(self, input):
        block1 = self.atrous_block1(input)
        block2 = self.atrous_block6(input)
        block3 = self.atrous_block12(input)
        block4 = self.atrous_block18(input)
        block5 = self.global_block(input)
        output = self.conv_1x1_output(torch.cat(block1, block2, block3, block4, block5), 2)
        return output


class aspp(nn.Module):
    def __init__(self,
                 module,
                 in_channels,
                 out_channels,
                 size):
        super(aspp, self).__init__()
        self.aspp_layers = []
        for i in range(len(in_channels)):
            aspp_layer = module(in_channels[i], out_channels[i], size[i])
            layer_name = 'layer{}'.format(i + 1)
            self.aspp_layers.append(layer_name)
            self.add_module(layer_name, aspp_layer)

    def forward(self, input):
        output = []
        for i, map in enumerate(input):
            aspp_layer = getattr(self, self.aspp_layers[i])
            output.append(aspp_layer(map))
        return output


