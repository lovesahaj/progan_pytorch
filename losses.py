import torch
import torch.nn as nn
from config import DEVICE, LAMBDA_GP


class WLossGP_Critic(nn.Module):
    def __init__(self, critic, λ=LAMBDA_GP):
        super(WLossGP_Critic, self).__init__()
        self.critic = critic
        self.λ = λ
        self.drift = 1e-3

    def __gp__(self, real, fake, α, train_step):
        batch, channels, height, width = real.shape

        ε = torch.randn((batch, 1, 1, 1)).repeat(
            (1, channels, height, width)).to(DEVICE)
        interpolated_image = real * ε + (fake * (1 - ε))
        mixed_score = self.critic(interpolated_image, α, train_step)

        gradient = torch.autograd.grad(
            inputs=interpolated_image,
            outputs=mixed_score,
            grad_outputs=torch.ones_like(mixed_score),
            retain_graph=True,
            create_graph=True)

        gradient = gradient[0].flatten(start_dim=1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1)**2)
        return penalty

    def forward(self, prediction_real, prediction_fake, real, fake, α,
                train_step):
        loss = torch.mean(prediction_fake) - torch.mean(prediction_real)
        gp = self.__gp__(real, fake, α, train_step)

        loss = loss + (gp * self.λ) + (self.drift *
                                       torch.mean(prediction_real**2))
        return loss


class WLoss_Generator(nn.Module):
    def __init__(self):
        super(WLoss_Generator, self).__init__()

    def forward(self, output):
        return -torch.mean(output)
