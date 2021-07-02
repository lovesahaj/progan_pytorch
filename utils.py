import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from config import Z_DIM, DEVICE


def get_mean_std(loader):
    sum, sum_squared, num_batches = 0, 0, 0

    for data in tqdm(loader):
        data = data.cuda()
        sum += torch.mean(data, dim=[0, 2, 3])
        sum_squared += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = sum / num_batches
    mean_of_squared = sum_squared / num_batches

    std = torch.sqrt(mean_of_squared - mean**2)

    return mean, std


def generate_noise(z_dim, batch_size=None):
    if batch_size is None:
        return torch.randn(1, z_dim, 1, 1)
    return torch.randn(batch_size, z_dim, 1, 1)


def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake,
                        tensorboard_step):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)

    with torch.no_grad():
        img_grid_real = make_grid(real[:4], normalize=True)
        img_grid_fake = make_grid(fake[:4], normalize=True)

        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def generate_examples(gen, step, n=100):
    gen.eval()
    α = 1.0

    for i in tqdm(range(n)):
        with torch.no_grad():
            noise = generate_noise(Z_DIM).to(DEVICE)
            img = gen(noise, α, step)
            save_image(img * 0.5 + 0.5, f"saved_examples/img_{i}.png")

    gen.train()
