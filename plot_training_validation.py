import numpy as np
import matplotlib.pyplot as plt
from train_cnn import train_cnn


def show_results(history):
    accuracy = [res['train_acc'] for res in history]
    losses = [res['train_loss'] for res in history]
    val_accuracy = [res['val_acc'] for res in history]
    val_losses = [res['val_loss'] for res in history]

    print(np.arange(0, 20))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(np.arange(1, 21), losses, '-o', label='Training Loss')
    ax1.plot(np.arange(1, 21), val_losses, '-o', label='Validation Loss')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(np.arange(1, 21), 100 * np.array(accuracy),
             '-o', label='TrainingAccuracy')
    ax2.plot(np.arange(1, 21), 100 * np.array(val_accuracy),
             '-o', label='Validation Accuracy')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.show()
    plt.savefig('Training_vs_Validation.png', dpi=600)


if __name__ == '__main__':
    res, foldperf = train_cnn()
    show_results(res)
