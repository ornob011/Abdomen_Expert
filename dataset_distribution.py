import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optparse
from dataset_distribution import train_test_file


def dataset_distribution(train_df):
    sns.set_style("darkgrid")
    anchor_images = train_df['Anchor']
    labels = []

    count_covid = 0
    count_noncovid = 0
    count_cap = 0
    for i in anchor_images:
        y = np.unique(train_df[train_df['Anchor'] == str(i)]['Label'])
        if (y[0] == 1):
            labels.append("NonCOVID")
            count_noncovid += 1
        elif (y[0] == 2):
            labels.append("COVID")
            count_covid += 1
        elif (y[0] == 3):
            labels.append("CAP")
            count_cap += 1

    labels = np.array(labels)

    print("Covid Count = {0}\nNonCovid Count = {1}\nCommunity Acquired Pneumonia Count = {2}".format(
        count_covid, count_noncovid, count_cap))

    dataset = pd.DataFrame(labels, anchor_images)
    dataset = dataset.set_axis(
        [*dataset.columns[:-1], 'Label'], axis=1, inplace=False)

    plt.figure(figsize=(11.69, 8.27))
    ax = sns.countplot(x='Label', data=dataset, palette="Set2")


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
