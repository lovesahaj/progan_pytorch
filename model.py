from math import log2

import torch
import torch.nn as nn

from config import DEVICE

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


def fade_in(alpha, scaled, generated):
    return torch.tanh(((1 - alpha) * scaled) + (alpha * generated))


def MiniBatchStdDev(x):
    batch, channels, height, width = x.shape
    value = torch.std(x, dim=0).mean().repeat(batch, 1, height, width)
    return torch.cat([x, value], dim=1)


def generate_noise(z_dim):
    return torch.randn(1, z_dim, 1, 1)


class WSConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 gain=2,
                 sampling='down'):
        super(WSConv2d, self).__init__()

        if sampling == 'down':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding)

        elif sampling == 'up':
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size, stride, padding)

        self.scale = gain / ((in_channels * kernel_size ** 2) ** 0.5)
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        t = self.conv(x * self.scale)
        u = self.bias.view(1, self.bias.shape[0], 1, 1)
        return t + u


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = torch.scalar_tensor(1e-8)

    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_pixel_norm=True,
    ):
        super(ConvBlock, self).__init__()

        # 1st Section
        block = [
            WSConv2d(in_channels, out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        ]

        if use_pixel_norm:
            block.append(PixelNorm())
        block.append(nn.Dropout2d())

        # 2nd Section
        block.append(WSConv2d(out_channels, out_channels))
        block.append(nn.LeakyReLU(negative_slope=0.2))
        if use_pixel_norm:
            block.append(PixelNorm())
        block.append(nn.Dropout2d())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.initial = nn.Sequential(
            PixelNorm(),
            WSConv2d(z_dim,
                     in_channels,
                     kernel_size=4,
                     stride=1,
                     padding=0,
                     sampling='up'),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,
                     in_channels,
                     kernel_size=3,
                     stride=1,
                     padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.blocks = nn.ModuleList()

        for i in range(len(factors) - 1):
            temp_in = int(factors[i] * in_channels)
            temp_out = int(factors[i + 1] * in_channels)

            self.blocks.append(ConvBlock(temp_in, temp_out))

        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x, alpha, steps):
        out = self.initial(x)

        if steps == 0:
            return self.get_rgb(out)

        for step in range(steps):
            upscale = self.upsample(out)
            out = self.blocks[step](upscale)

        final_upscale = self.get_rgb(upscale)
        final_generated = self.get_rgb(out)

        out = fade_in(alpha, final_upscale, final_generated)
        return out

    def save(self):
        torch.save(self.state_dict(), 'generator.pth.tar')

    def load(self):
        c = torch.load('generator.pth.tar', map_location=DEVICE)
        self.load_state_dict(c)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight.data)

    def get_rgb(self, x):
        layer = WSConv2d(x.shape[1],
                         self.img_channels,
                         kernel_size=1,
                         stride=1, padding=0).to(DEVICE)
        return layer(x)


class Critic(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Critic, self).__init__()
        self.img_channels = img_channels
        self.blocks = nn.ModuleList()
        self.out_channels = []

        for i in range((len(factors) - 1), 0, -1):
            temp_in = int(in_channels * factors[i])
            temp_out = int(in_channels * factors[i - 1])

            self.blocks.append(
                ConvBlock(temp_in, temp_out, use_pixel_norm=False))
            self.out_channels.append(temp_in)

        self.out_channels.append(in_channels)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1,
                     in_channels,
                     kernel_size=3,
                     padding=1,
                     stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,
                     in_channels,
                     kernel_size=4,
                     padding=0,
                     stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, alpha, steps):
        current_step = len(self.blocks) - steps
        out = self.leaky(self.from_rgb(x, self.out_channels[current_step]))

        if steps == 0:
            out = MiniBatchStdDev(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(
            self.from_rgb(self.downsample(x),
                          self.out_channels[current_step + 1]))

        out = self.blocks[current_step](out)
        out = self.downsample(out)

        out = fade_in(alpha, downscaled, out)

        for step in range(current_step + 1, len(self.blocks)):
            out = self.downsample(self.blocks[step](out))

        out = MiniBatchStdDev(out)
        out = self.final_block(out).view(out.shape[0], -1)

        return out

    def save(self):
        torch.save(self.state_dict(), 'generator.pth.tar')

    def load(self):
        c = torch.load('generator.pth.tar', map_location=DEVICE)
        self.load_state_dict(c)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight.data)

    def from_rgb(self, x, out_channels):
        layer = WSConv2d(self.img_channels,
                         out_channels,
                         kernel_size=1,
                         stride=1, padding=0).to(DEVICE)
        return layer(x)


def test():
    z_dim = 512
    in_channels = 256
    img_channels = 3

    gen = Generator(z_dim, in_channels, img_channels)
    gen = gen.cuda()
    gen.init_weights()
    crit = Critic(in_channels, img_channels)
    crit = crit.cuda()
    crit.init_weights()

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_step = int(log2(img_size / 4))
        x = generate_noise(z_dim).cuda()
        z = gen(x, 0.5, steps=num_step)

        assert z.shape == (1, 3, img_size, img_size)

        out = crit(z, 0.5, steps=num_step)
        assert out.shape == (1, 1)

        print(f"Success!! At img size: {img_size}")


if __name__ == '__main__':
    test()
