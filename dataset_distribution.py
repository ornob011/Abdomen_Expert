import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optparse
from dataset_distribution import train_test_file


def dataset_distribution_for_training(train_df):
#     sns.set_style("darkgrid")
#     images = train_df['ID']
    images = []
    label_cols = ['Kidney Cyst', 'Kidney Stone', 'Kidney Tumour',
                   'Liver Tumour', 'Adenocarcinoma (Lung Cancer)', 'Benign (Lung Cancer)',
                   'Large Carcinoma (Lung Cancer)', 'Squamous Carcinoma (Lung Cancer)', 'Stomach Cancer',
                   'Normal']
    
    labels = []
    
    for i in label_cols:
        count = train_df[i].sum()
        labels.append(count)
    
    dataset = {'Kidney Cyst':labels[0], 'Kidney Stone':labels[1], 'Kidney Tumour':labels[2],
            'Liver Tumour':labels[3], 'Adenocarcinoma (Lung Cancer)':labels[4], 'Benign (Lung Cancer)':labels[5], 
            'Large Carcinoma (Lung Cancer)':labels[6], 'Squamous Carcinoma (Lung Cancer)':labels[7], 'Stomach Cancer':labels[8], 
            'Normal':labels[9]}
    
    classes = list(dataset.keys())
    values = list(dataset.values())
    
    dataset = pd.DataFrame(dataset, index=[0])
    
    plt.figure(figsize=(20,10))
    sns.barplot(data=dataset)  
    

if __name__ == '__main__':

    parser = optparse.OptionParser()

    parser.add_option('-t', '--distribution', action="store", dest="distribution",
                      help="Input train or test", default="--train_distribution")

    options, args = parser.parse_args()

    distribution = str(options.distribution)

    train_df, test_df = train_test_file()
    if (distribution == 'train'):
        dataset_distribution(train_df)

    elif (distribution == 'test'):
        dataset_distribution(test_df)
    else:
        print("Wrong Input\n")
