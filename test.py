from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.functional import InterpolationMode
from dataset import FFHQ
from utils import get_mean_std


def main():
    # dataset = AnimeDataset("./images/*.jpg", (70, 70))
    tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70),
                                      interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor()
    ])
    dataset = FFHQ('data/archive/*.png')
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12)

    print(get_mean_std(loader))


if __name__ == '__main__':
    main()
