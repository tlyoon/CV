import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,title, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
#    print(cm)
#    print('')
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 3.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    #print('print cm from pltcmtrx.py')
    #plt.show()
    #pngtt=0.KNeighborsClassifier(); a_scr=1.000; geo_mean=1.000
    #print('from pltcmtrx.py: title=',title)
    if title.count('Classifier')==1:
        dumtitle=title.split(';')[0].split('Classifier')
        pngtt=dumtitle[0]+dumtitle[1]
    else:
        pngtt=title.split(';')[0]
    
#    pngtt=title.split(';')[0].split('Classifier')[0]
    #print('from pltcmtrx.py: pngtt=',pngtt)
    print(str(pngtt)+'.png saved')
    #print(' ')
    #pngtt=title
    fig.savefig(str(pngtt)+'.png') # This is just to show the figure is still generated
    return fig