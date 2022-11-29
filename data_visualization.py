import pandas as pd
import cv2
import matplotlib.pyplot as plt
import optparse

def show_image(index, DATA_DIR, CSV_FILE, df, dim, declasses):             
    row = df.iloc[index]

    A_label = row.Label

    N_img = cv2.imread(DATA_DIR + row.Negative)
    N_img = cv2.cvtColor(N_img, cv2.COLOR_BGR2RGB) 
    N_img = cv2.resize(N_img, dim, interpolation = cv2.INTER_AREA)

    A_img = cv2.imread(DATA_DIR + row.Anchor)
    A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB) 
    A_img = cv2.resize(A_img, dim, interpolation = cv2.INTER_AREA)

    P_img = cv2.imread(DATA_DIR + row.Positive)
    P_img = cv2.cvtColor(P_img, cv2.COLOR_BGR2RGB)
    A_img = cv2.resize(A_img, dim, interpolation = cv2.INTER_AREA)


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 17))

    ax1.set_title("Anchor: "+ declasses[A_label])
    ax1.imshow(A_img)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])

    ax2.set_title("Positive: " + declasses[A_label])
    ax2.imshow(P_img)
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])

    ax3.set_title("Negative")
    ax3.imshow(N_img)
    ax3.axes.xaxis.set_ticklabels([])
    ax3.axes.yaxis.set_ticklabels([])
    

if __name__ == '__main__':

    parser = optparse.OptionParser()

    parser.add_option('-i', '--index', action="store", dest="index", help="Input a random number", default="100")
    options, args = parser.parse_args()
    index = int(options.index)

    DATA_DIR = 'main/'
    CSV_FILE = 'trip.csv'

    df = pd.read_csv(CSV_FILE)
    dim = (224, 224)

    declasses = {1: 'Kidney Cyst', 2: 'Kidney Stone', 3: 'Kidney Tumour', 
                4: 'Liver Tumour', 5: 'Adenocarcinoma (Lung Cancer)', 6: 'Benign (Lung Cancer)',
                7: 'Large Carcinoma (Lung Cancer)', 8: 'Squamous Carcinoma (Lung Cancer)', 9: 'Stomach Cancer',
                10: 'Normal'}

    show_image(index, DATA_DIR, CSV_FILE, df, dim, declasses)