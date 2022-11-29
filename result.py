import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.io as pio
from sklearn.metrics import roc_auc_score, hamming_loss, cohen_kappa_score, log_loss, roc_curve, auc, precision_recall_curve, average_precision_score, PrecisionRecallDisplay, confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff


def plotly_confusion_matrix():

    z = cm
    z = z[::-1]
    x = class_names
    y = x[::-1].copy()

    z_text = [[str(y) for y in x] for x in z]

    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, annotation_text=z_text, colorscale='Blues')

    fig.update_layout(title_text='<i><b>Confusion Matrix</b></i>',)

    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.1,
                            showarrow=False,
                            text="Predicted label",
                            xref="paper",
                            yref="paper"))

    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.55,
                            y=0.5,
                            showarrow=False,
                            text="True label",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    fig.update_layout(title_x=.5, title_y=0.999999,  width=700, height=600)

    fig.show()


def receiver_operating_curve():
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range(z_prob.shape[1]):
        z_lb_ = z_lb_true[:, i]
        z_ = z_prob[:, i]

        fpr, tpr, _ = roc_curve(z_lb_, z_)
        auc_score = roc_auc_score(z_lb_, z_)

        name = f"{class_names[i]} (AUC={auc_score:.3f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=650,
        margin=dict(l=3, r=3, b=30, t=30, pad=4),
        title='Receiver operating characteristic curve for all classes',
        title_x=.5

    )
    fig.show()


def print_confusion_matrix(confusion_matrix, class_names, figsize=(8.4, 6.8), fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])

        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) /\
                float(len(set_true.union(set_pred)))

        acc_list.append(tmp_a)
    return np.mean(acc_list)


if __name__ == '__main__':
    
    z_true = np.load('z_true.npy')
    z_pred = np.load('z_pred.npy')
    z_lb_true = np.load('z_lb_true.npy')
    z_lb_pred = np.load('z_lb_pred.npy')
    z_prob = np.load('z_prob.npy')

    class_names = ['Kidney Cyst', 'Kidney Stone', 'Kidney Tumour',
                   'Liver Tumour', 'Adenocarcinoma (Lung Cancer)', 'Benign (Lung Cancer)',
                   'Large Carcinoma (Lung Cancer)', 'Squamous Carcinoma (Lung Cancer)', 'Stomach Cancer',
                   'Normal']

    cm = confusion_matrix(z_true, z_pred)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print('Overall Accuracy:  {:.4f}'.format(accuracy_score(z_true, z_pred)))

    print('Hamming score: {:.4f}'.format(hamming_score(z_true, z_pred)))

    print('Hamming loss: {:.4f}'.format(hamming_loss(z_true, z_pred)))

    print('Cohen Kappa Score: {:.4f}'.format(
        cohen_kappa_score(z_true, z_pred)))

    print('Log Loss: {:.4f}'.format(log_loss(z_lb_true, z_lb_pred)))

    print('Micro Avg. ROC AUC Score: {:.4f}'.format(
        roc_auc_score(z_lb_true, z_lb_pred, average='micro')))

    print('Macro Avg. ROC AUC Score: {:.4f}'.format(
        roc_auc_score(z_lb_true, z_lb_pred, average='macro')))

    print('\nClassification Report:\n')
    print(classification_report(z_lb_true, z_lb_pred, target_names=class_names))

    print('\nSpecificity Report:\n')
    for i, n in enumerate(class_names):
        print(class_names[i], "%.4f" % TNR[i])

    print('\nAccuray Report:\n')
    for i, n in enumerate(class_names):
        print(class_names[i], "%.4f" % ACC[i])

    print('\nAUC Score:\n')
    for i, n in enumerate(class_names):
        print(class_names[i], "%.4f" %
              roc_auc_score(z_lb_true[:, i], z_lb_pred[:, i]))

    receiver_operating_curve()

    plotly_confusion_matrix()
