from sklearn.metrics import confusion_matrix
from operator import truediv
import matplotlib.pylab as plt
import numpy as np


def plot_confusion_matrix(cm, cmap=plt.cm.Blues):
    classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                   'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                   'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                   'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                   'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    plt.rcParams['savefig.dpi'] = 300  #图片像素
    plt.rcParams['figure.dpi'] = 300  #分辨率
    # plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    plt.tick_params(labelsize=5)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes)
    plt.title('Confusion matrix', fontsize=6)
    plt.xlabel('Predicted label', fontsize=6)
    plt.ylabel('True label', fontsize=6)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", fontsize=3,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('conf_mtx.png')


def record_output(oa, aa, each_acc, path):
    with open(path, 'w') as f:
        sentence0 = 'OA :' + str(oa) + '\n'
        f.write(sentence0)
        sentence1 = 'AA :' + str(aa) + '\n'
        f.write(sentence1)
        sentence2 = "Acc for each classes : " + str(each_acc) + '\n\n'
        f.write(sentence2)


def record_times(time1, time2, path):
    with open(path, 'a') as f:
        sentence3 = 'The training time of first stage: %d m' % time1 + '\n'
        f.write(sentence3)
        sentence4 = 'The training time of second stage:%d m' % time2 + '\n'
        f.write(sentence4)
