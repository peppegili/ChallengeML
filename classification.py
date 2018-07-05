#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.autograd import Variable
import time
import copy
from matplotlib import pyplot as plt
import numpy as np
import itertools





# ------- TRAINING PROCEDURE -------

def train_classification(model, train_loader, test_loader, lr=0.00001, epochs=100, momentum=0.9):
    since = time.time()
    print "---- TRAINING START ----"

    best_model = copy.deepcopy(model.state_dict())  # for best model
    best_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=momentum)

    loaders = {'train': train_loader, 'test': test_loader}
    losses = {'train': [], 'test': []}
    accuracies = {'train': [], 'test': []}

    if torch.cuda.is_available():
        model = model.cuda()

    for e in range(epochs):
        for mode in ['train', 'test']:
            if mode == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            epoch_acc = 0
            samples = 0
            for i, batch in enumerate(loaders[mode]):
                # trasformiamo i tensori in variabili
                x = Variable(batch['image'], requires_grad=(mode == 'train'))
                y = Variable(batch['label'])

                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()

                output = model(x)
                l = criterion(output, y)

                if mode == 'train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                acc = accuracy_score(y.data, output.max(1)[1].data)

                epoch_loss += l.data[0] * x.shape[0]
                epoch_acc += acc * x.shape[0]
                samples += x.shape[0]

                print "\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" %\
                      (mode, e + 1, epochs, i, len(loaders[mode]), epoch_loss / samples, epoch_acc / samples),

            epoch_loss /= samples
            epoch_acc /= samples

            losses[mode].append(epoch_loss)
            accuracies[mode].append(epoch_acc)

            print "\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" %\
                  (mode, e + 1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc)

            # deep copy the model
            if mode == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

    print""

    time_elapsed = time.time() - since
    print('---- TRAINING COMPLETE in {:.0f}m {:.0f}s ----'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model)
    # torch.save(model.state_dict(),'classification_model.pth')

    # restituiamo il modello e i vari log
    return model, (losses, accuracies)






#----- PLOT LOGS--------

def plot_logs_classification(logs):
    training_losses, training_accuracies, test_losses, test_accuracies =\
        logs[0]['train'], logs[1]['train'], logs[0]['test'], logs[1]['test']
    plt.figure(figsize=(18,6))
    plt.subplot(121)
    plt.plot(training_losses)
    plt.plot(test_losses)
    plt.legend(['Training Loss','Test Losses'])
    plt.grid()
    plt.subplot(122)
    plt.plot(training_accuracies)
    plt.plot(test_accuracies)
    plt.legend(['Training Accuracy','Test Accuracy'])
    plt.grid()
    #plt.show()




# ----- TEST MODEL -------

def test_model_classification(model, test_loader):
    softmax = nn.Softmax(dim=1)
    model.eval()
    preds = []
    gts = []
    for batch in test_loader:
        x=Variable(batch['image'])
        #applichiamo la funzione softmax per avere delle probabilità
        if torch.cuda.is_available():
            x = x.cuda()
        pred = softmax(model(x)).data.cpu().numpy().copy()
        gt = batch['label'].cpu().numpy().copy()
        preds.append(pred)
        gts.append(gt)
    return np.concatenate(preds),np.concatenate(gts)





# ----- ACCURACY --------





# ---- CONFUSION MATRIX ----

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')





# ---- SCORE F1 AND mF1----








# ---- NORMALIZATION ------

def normalization(train):
    m = np.zeros(3)  # 3 canali

    for sample in train:
        m += sample['image'].sum(1).sum(1)  # accumuliamo la somma dei pixel canale per canale
    # dividiamo per il numero di immagini moltiplicato per il numero di pixel
    m = m / (len(train) * 144 * 256)

    # procedura simile per calcolare la deviazione standard
    s = np.zeros(3)

    for sample in train:
        s += ((sample['image'] - torch.Tensor(m).view(3, 1, 1)) ** 2).sum(1).sum(1)
    s = np.sqrt(s / (len(train) * 144 * 256))

    # media e devST per i 3 canali
    # print "Medie",m
    # print "Dev.Std.",s
    return m,s