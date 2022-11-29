from split_dataset import train_test_file
import numpy as np
import torch
import cv2
from tqdm.notebook import tqdm
import pandas as pd
from get_model_optimizer import get_model, get_optimizer


def get_encoding_csv(model, anc_img_names, labels):
    anc_img_names_arr = np.array(anc_img_names)
    labels = np.array(labels)
    encodings = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(anc_img_names_arr):
            A = cv2.imread(DATA_DIR + i)
            A = cv2.resize(A, dsize=(size, size),
                           interpolation=cv2.INTER_CUBIC)
            A = torch.from_numpy(A).permute(2, 0, 1)/255.0
            A = A.to(DEVICE)
            A_enc = model(A.unsqueeze(0)).to(DEVICE)
            encodings.append(A_enc.squeeze().cpu().detach().numpy())

        encodings = np.array(encodings)
        labels = pd.DataFrame(labels)
        encodings = pd.DataFrame(encodings)
        df_enc = pd.concat([anc_img_names, encodings, labels], axis=1)

    return df_enc


if __name__ == '__main__':

    df, test_df = train_test_file()
    DATA_DIR = 'main/'
    size = 224
    DEVICE = 'cuda'

    model = get_model()
    checkpoint = torch.load('proposed_model_all.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    df_enc = get_encoding_csv(model, df['Anchor'], df['Label'])

    df_enc = df_enc.set_axis(
        [*df_enc.columns[:-1], 'Label'], axis=1, inplace=False)
    df_enc.to_csv('proposed_model_encoding.csv', index=False)
