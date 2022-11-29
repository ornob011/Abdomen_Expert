import pandas as pd
import numpy as np
import cv2
import torch
from split_dataset import train_test_file
from get_model_optimizer import get_model
from torch import nn
from tqdm.notebook import tqdm
from sklearn import preprocessing


def eucliedean_dist(img_enc, anc_enc_arr):
    dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T))
    return dist


def get_prediction_variables(idx):

    img_name = test_df['Anchor'].iloc[idx]
    img_path = DATA_DIR + img_name

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    img = torch.from_numpy(img).permute(2, 0, 1)/255.0

    model.eval()
    with torch.no_grad():
        img = img.to(DEVICE)
        img_enc = model(img.unsqueeze(0)).to(DEVICE)
        img_enc = img_enc.detach().cpu().numpy()

    anc_enc_arr = df_enc.iloc[:, 1:513].to_numpy()
    anc_img_names = df_enc['Anchor']

    distance = []

    for i in range(anc_enc_arr.shape[0]):
        dist = eucliedean_dist(img_enc, anc_enc_arr[i: i+1, :])

        distance = np.append(distance, dist)

    closest_idx = np.argsort(distance)

    return anc_img_names, img, img_path, closest_idx, distance, img_enc


def get_label(i, img_enc, anc_img_names, DATA_DIR, img, img_path, closest_idx, no_of_closest=1):

    img_name = img_path.split('/')[-1]
    S_name = []
    temp1 = np.unique(whole_df[whole_df['Anchor'] == img_name]['Label'])

    z_true.append(temp1[0])

    for s in range(no_of_closest):
        S_name.append(anc_img_names.iloc[closest_idx[s]])

    wanted_label = list(set(S_name))

    res_name = wanted_label[0]

    temp2 = np.unique(whole_df[whole_df['Anchor'] == res_name]['Label'])

    z_pred.append(temp2[0])

    print(temp1[0], temp2[0])

    if (temp2[0] == 3):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][2] = similarity[0]

        z_prob[i][0] = dis

        z_prob[i][1] = dis
        z_prob[i][3] = dis
        z_prob[i][4] = dis
        z_prob[i][5] = dis
        z_prob[i][6] = dis
        z_prob[i][7] = dis
        z_prob[i][8] = dis
        z_prob[i][9] = dis


#         print(similarity[0], dis, dis)

    elif (temp2[0] == 2):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][1] = similarity[0]

        z_prob[i][0] = dis

        z_prob[i][2] = dis
        z_prob[i][3] = dis
        z_prob[i][4] = dis
        z_prob[i][5] = dis
        z_prob[i][6] = dis
        z_prob[i][7] = dis
        z_prob[i][8] = dis
        z_prob[i][9] = dis

#         print(similarity[0], dis, dis)

    elif (temp2[0] == 1):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][0] = similarity[0]

        z_prob[i][2] = dis
        z_prob[i][1] = dis
        z_prob[i][3] = dis
        z_prob[i][4] = dis
        z_prob[i][5] = dis
        z_prob[i][6] = dis
        z_prob[i][7] = dis
        z_prob[i][8] = dis
        z_prob[i][9] = dis

    elif (temp2[0] == 4):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][3] = similarity[0]

        z_prob[i][0] = dis

        z_prob[i][1] = dis
        z_prob[i][2] = dis
        z_prob[i][4] = dis
        z_prob[i][5] = dis
        z_prob[i][6] = dis
        z_prob[i][7] = dis
        z_prob[i][8] = dis
        z_prob[i][9] = dis

    elif (temp2[0] == 5):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][4] = similarity[0]

        z_prob[i][0] = dis

        z_prob[i][1] = dis
        z_prob[i][2] = dis
        z_prob[i][3] = dis
        z_prob[i][5] = dis
        z_prob[i][6] = dis
        z_prob[i][7] = dis
        z_prob[i][8] = dis
        z_prob[i][9] = dis

    elif (temp2[0] == 6):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][5] = similarity[0]

        z_prob[i][0] = dis

        z_prob[i][1] = dis
        z_prob[i][2] = dis
        z_prob[i][3] = dis
        z_prob[i][4] = dis
        z_prob[i][6] = dis
        z_prob[i][7] = dis
        z_prob[i][8] = dis
        z_prob[i][9] = dis

    elif (temp2[0] == 7):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][6] = similarity[0]

        z_prob[i][0] = dis

        z_prob[i][1] = dis
        z_prob[i][2] = dis
        z_prob[i][3] = dis
        z_prob[i][4] = dis
        z_prob[i][5] = dis
        z_prob[i][7] = dis
        z_prob[i][8] = dis
        z_prob[i][9] = dis

    elif (temp2[0] == 8):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][7] = similarity[0]

        z_prob[i][0] = dis

        z_prob[i][1] = dis
        z_prob[i][2] = dis
        z_prob[i][3] = dis
        z_prob[i][4] = dis
        z_prob[i][5] = dis
        z_prob[i][6] = dis
        z_prob[i][8] = dis
        z_prob[i][9] = dis

    elif (temp2[0] == 9):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][8] = similarity[0]

        z_prob[i][0] = dis

        z_prob[i][1] = dis
        z_prob[i][2] = dis
        z_prob[i][3] = dis
        z_prob[i][4] = dis
        z_prob[i][5] = dis
        z_prob[i][6] = dis
        z_prob[i][7] = dis
        z_prob[i][9] = dis

    elif (temp2[0] == 10):

        similar_enc = np.array(df_enc[df_enc['Anchor'] == res_name])
        similar_enc = np.array(similar_enc[:, 1:513])
        similar_enc = np.vstack(similar_enc).astype(float)

        img_enc = torch.from_numpy(img_enc)
        similar_enc = torch.from_numpy(similar_enc)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity = cos(img_enc, similar_enc)

        dis = (1-similarity[0])/2

        z_prob[i][9] = similarity[0]

        z_prob[i][0] = dis

        z_prob[i][1] = dis
        z_prob[i][2] = dis
        z_prob[i][3] = dis
        z_prob[i][4] = dis
        z_prob[i][5] = dis
        z_prob[i][6] = dis
        z_prob[i][7] = dis
        z_prob[i][8] = dis


#         print(similarity[0], dis, dis)

    return z_true, z_pred, z_prob


if __name__ == '__main__':

    train_df, test_df = train_test_file()
    DATA_DIR = 'main/'
    size = 224
    DEVICE = 'cuda'

    model = get_model()
    checkpoint = torch.load('proposed_model_all.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    df_enc = pd.read_csv('proposed_model_encoding.csv')

    class_names = ['Kidney Cyst', 'Kidney Stone', 'Kidney Tumour',
                   'Liver Tumour', 'Adenocarcinoma (Lung Cancer)', 'Benign (Lung Cancer)',
                   'Large Carcinoma (Lung Cancer)', 'Squamous Carcinoma (Lung Cancer)', 'Stomach Cancer',
                   'Normal']

    whole_df = pd.read_csv('full_dataset.csv')

    z_true = []
    z_pred = []

    s = (test_df.shape[0], 10)
    z_prob = np.ones(s)

    for i in tqdm(range(test_df.shape[0])):
        anc_img_names, img, img_path, closest_idx, distance, img_enc = get_prediction_variables(
            i)
        z_true, z_pred, z_prob = get_label(
            i, img_enc, anc_img_names, DATA_DIR, img, img_path, closest_idx)

    z_true = np.array(z_true)
    z_pred = np.array(z_pred)

    z_prob = np.array(z_prob)

    lb = preprocessing.LabelBinarizer()
    lb.fit(z_true)

    z_lb_true = lb.transform(z_true)
    z_lb_pred = lb.transform(z_pred)

    np.save('z_true_final.npy', z_true)
    np.save('z_pred_final.npy', z_pred)
    np.save('z_lb_true_final.npy', z_lb_true)
    np.save('z_lb_pred_final.npy', z_lb_pred)
    np.save('z_prob_final.npy', z_prob)
