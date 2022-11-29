from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T
from split_dataset import train_test_file


def get_dataloader():

    DATA_DIR = 'main/'

    class APN_Dataset(Dataset):
        def __init__(self, df, transform=None):
            self.df = df
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):

            row = self.df.iloc[idx]
            A_img = Image.open(os.path.join(
                DATA_DIR, row.Anchor)).convert("RGB")
            P_img = Image.open(os.path.join(
                DATA_DIR, row.Positive)).convert("RGB")
            N_img = Image.open(os.path.join(
                DATA_DIR, row.Negative)).convert("RGB")

            label = row.Label

            if self.transform:
                A_img = self.transform(A_img)
                P_img = self.transform(P_img)
                N_img = self.transform(N_img)

            return A_img, P_img, N_img, label

    ImageNet_Mean_Std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    size = 224

    transformer_train = T.Compose([
        T.Resize(size),
        T.RandomVerticalFlip(0.4),
        T.RandomHorizontalFlip(0.4),
        T.RandomAdjustSharpness(sharpness_factor=2),
        T.RandomAutocontrast(),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(*ImageNet_Mean_Std)
    ])

    transformer_test = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(*ImageNet_Mean_Std)
    ])

    train_df, test_df = train_test_file()
    trainset = APN_Dataset(train_df, transformer_test)

    return trainset
