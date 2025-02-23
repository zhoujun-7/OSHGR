import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_confusion_matrix(tg: np.ndarray, pd: np.ndarray, save_path=None):
    # pd: c
    # tg: c,
    conf_mat = confusion_matrix(tg, pd, normalize="pred")
    conf_mat = conf_mat * 100
    conf_mat = conf_mat.astype(np.int32)
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1)
    disp = ConfusionMatrixDisplay(conf_mat)
    disp.plot(ax=ax, values_format="3d")
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
        plt.close()


if __name__ == "__main__":
    pd = np.random.randint(0, 68, [1000])
    tg = np.random.randint(0, 68, [1000])
    print(pd)
    get_confusion_matrix(pd, tg, "qqq.jpg")
