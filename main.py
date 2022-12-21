import numpy as np
import matplotlib.pyplot as plt
from train_cnn import train_cnn
from plot_training_validation import show_results

if __name__ == '__main__':
    res, foldperf = train_cnn()
    show_results(res)
