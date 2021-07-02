from config import BATCH_SIZES, NUM_WORKER, PIN_MEMORY
import albumentations as A
from torch.utils.data import DataLoader, Dataset, random_split
from glob import glob
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt


class FFHQ(Dataset):
    """
    MEAN: [0.5205, 0.4254, 0.3804]
    STDDEV: [0.2825, 0.2567, 0.2575]
    MEMORY ON GPU: 1297 MB (Batch Size -> 64)
    """

    def __init__(self, dir, size=None, transforms=ToTensorV2()) -> None:
        super(FFHQ, self).__init__()
        self.img_files = glob(dir)
        self.size = size
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.size is not None:
            img = cv2.resize(img, self.size)

        img = self.transforms(image=img)["image"]

        return (img / 255)

    def dataloader(self, **kwargs):
        loader = DataLoader(self, **kwargs)
        return loader

    def train_test_split(self, test_train_ratio):
        assert test_train_ratio != 0

        no_test = int(len(self) * test_train_ratio)
        train, test = random_split(self, [len(self) - no_test, no_test])

        return train, test


def test():
    transforms = A.Compose([
        A.transforms.Normalize(mean=(0.5205, 0.4254, 0.3804),
                               std=(0.2825, 0.2567, 0.2575)),
        ToTensorV2()
    ])
    dataset = FFHQ('data/archive/*.png')

    loader = dataset.dataloader(batch_size=BATCH_SIZES[0],
                                num_workers=NUM_WORKER,
                                pin_memory=PIN_MEMORY)

    print(len(dataset))

    img = next(iter(loader))[0]
    print(img.shape)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    test()
