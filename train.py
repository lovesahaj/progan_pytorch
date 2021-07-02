from utils import generate_examples, generate_noise, plot_to_tensorboard
from tqdm import tqdm
from dataset import FFHQ
from math import log2
import torch.cuda.amp
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from models import Generator, Critic
from config import (BATCH_SIZES, DEVICE, FIXED_NOISE, IMG_CHANNELS,
                    IN_CHANNELS, LAMBDA_GP, LRN_RATE, ADAM_BETAS, LOAD_MODEL,
                    NUM_WORKER, PIN_MEMORY, PROGRESSIVE_EPOCHS, SAVE_MODEL,
                    START_TRAINING_AT_IMG_SIZE, Z_DIM)
from losses import WLossGP_Critic, WLoss_Generator


class TrainModel(nn.Module):
    def __init__(self, gen, critic, device):
        super(TrainModel, self).__init__()
        self.device = device

        self.gen = gen.to(device)
        self.critic = critic.to(device)

        self.gen_opt = optim.Adam(gen.parameters(),
                                  lr=LRN_RATE,
                                  betas=ADAM_BETAS)
        self.critic_opt = optim.Adam(critic.parameters(),
                                     lr=LRN_RATE,
                                     betas=ADAM_BETAS)

        self.critic_loss_fn = WLossGP_Critic(self.critic, LAMBDA_GP)
        self.gen_loss_fn = WLoss_Generator()

        self.scaler_critic = torch.cuda.amp.GradScaler()
        self.scaler_gen = torch.cuda.amp.GradScaler()

        self.writer = SummaryWriter("logs/gan1")
        self.tensorboard_step = 0

        if LOAD_MODEL:
            self.__load_checkpoint__()

        self.gen.init_weights()
        self.critic.init_weights()

        self.gen.train()
        self.critic.train()

    def __train_one__(self, alpha, loader, step, len_dataset):
        loop = tqdm(loader, leave=True)

        for batch_idx, real in enumerate(loop):
            cur_batch_size = real.shape[0]
            real = real.to(self.device)

            noise = generate_noise(Z_DIM, cur_batch_size).to(self.device)

            # with torch.cuda.amp.autocast():
            fake = self.gen(noise, alpha, step)
            # print(fake)
            # exit()
            pred_r = self.critic(real, alpha, step)
            pred_f = self.critic(fake.detach(), alpha, step)

            loss_c = self.critic_loss_fn(prediction_real=pred_r,
                                         prediction_fake=pred_f,
                                         real=real,
                                         fake=fake,
                                         α=alpha,
                                         train_step=step)

            self.critic_opt.zero_grad()
            self.scaler_critic.scale(loss_c).backward(retain_graph=True)
            self.scaler_critic.step(self.critic_opt)
            self.scaler_critic.update()

            # with torch.cuda.amp.autocast():
            gen_fake = self.critic(fake, alpha, step)
            loss_g = self.gen_loss_fn(gen_fake)

            self.gen_opt.zero_grad()
            self.scaler_gen.scale(loss_g).backward()
            self.scaler_gen.step(self.gen_opt)
            self.scaler_gen.update()

            alpha += cur_batch_size / (len_dataset * PROGRESSIVE_EPOCHS[step] *
                                       0.5)
            alpha = min(alpha, 1)

            if batch_idx % 5 == 0:
                with torch.no_grad():
                    fixed_fakes = self.gen(FIXED_NOISE, alpha,
                                           step) * 0.5 + 0.5
                    plot_to_tensorboard(self.writer, loss_c.item(),
                                        loss_g.item(), real.detach(),
                                        fixed_fakes.detach(),
                                        self.tensorboard_step)
                    self.tensorboard_step += 1

            loop.set_postfix(loss_critic=loss_c.item(), loss_gen=loss_g.item())

        return alpha

    def forward(self, alpha, loader, num_epochs, step, len_dataset):
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1} / {num_epochs}]")
            alpha = self.__train_one__(alpha, loader, step, len_dataset)
            generate_examples(self.gen, step)

        if SAVE_MODEL:
            self.__save_checkpoint__()

        step += 1

    def __load_checkpoint__(self):
        print("=>> Loading Checkpoint")
        self.gen.load(self.gen_opt)
        self.critic.load(self.critic_opt)

    def __save_checkpoint__(self):
        print("=>> Saving Checkpoint")
        self.gen.save(self.gen_opt)
        self.critic.save(self.critic_opt)


def main():
    gen = Generator(Z_DIM, IN_CHANNELS, IMG_CHANNELS)
    critic = Critic(IN_CHANNELS, IMG_CHANNELS)

    transforms = A.Compose([
        A.transforms.HorizontalFlip(p=0.5),
        A.transforms.Normalize(mean=(0.5205, 0.4254, 0.3804),
                               std=(0.2825, 0.2567, 0.2575)),
        ToTensorV2()
    ])

    trainObj = TrainModel(gen, critic, DEVICE)

    step = int(log2(START_TRAINING_AT_IMG_SIZE / 4))

    for num_epoch in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        img_size = 4 * 2**step
        dataset = FFHQ('data/archive/*.png',
                       size=(img_size, img_size),
                       transforms=transforms)

        loader = dataset.dataloader(batch_size=BATCH_SIZES[step],
                                    num_workers=NUM_WORKER,
                                    pin_memory=PIN_MEMORY)

        print(f"Image Size: {img_size}")
        trainObj(alpha, loader, num_epoch, step, len(dataset))


if __name__ == "__main__":
    main()