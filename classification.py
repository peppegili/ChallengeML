#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.autograd import Variable
import time
import copy
from matplotlib import pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import Dataset
from classification_models import *



#-----------------------------------------------------------------------------------------

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



#------------------------------------------------------------------------------------------------





#---------------------------------- NEURAL NETWORKS -----------------------------------------

def mlp():

    # ------ MODEL -------
    mlp_classifier = MLPClassifier(110592, 16, 512) # input, classi, hidden neurons


    # training set
    train = Dataset('dataset/images','dataset/training_list.csv',transform=transforms.ToTensor())



    #------ NORMALIZATION ---------

    m,s = normalization(train)




    # --------- DATALOADER AND TRANSFORMATION ----------

    transform = transforms.Compose([transforms.ToTensor(), #conversione in tensore
                                    transforms.Normalize(m,s), # nomrlizzazione con media e dvst del TS
                                    transforms.Lambda(lambda x: x.view(-1))]) # trasforma l'immagine in un unico vettore

    #ridefiniamo il training set specificando le trasformazioni
    train = Dataset('dataset/images','dataset/training_list.csv',transform=transform)

    # permutiamo i dati di training
    idx = np.random.permutation(len(train))
    train.images = train.images[idx]

    #print "Immagine di train:", train[0]['image'].shape # 3x144x256
    #print "Etichetta:", train[0]['label']
    #print "Posa:", train[0]['pose']



    #print ""


    #ridefiniamo il test set specificando le trasformazioni
    valid = Dataset('dataset/images','dataset/validation_list.csv',transform=transform)

    # permutiamo i dati di validation
    idx = np.random.permutation(len(valid))
    valid.images = valid.images[idx]

    #print "Immagine di validation:", valid[0]['image'].shape # 3x144x256
    #print "Etichetta:", valid[0]['label']
    #print "Posa:", valid[0]['pose']


    #definiamo i dataloaders
    train_loader = DataLoader(train, batch_size=64, num_workers=2, shuffle=True) #shuffle accede ai dati in maniera casuale
    #shuffle permette di accedere ai dati in maniera casuale
    valid_loader = DataLoader(valid, batch_size=64, num_workers=2)





    #----- START TRAINING -------

    mlp_classifier, mlp_classifier_logs = train_classification(mlp_classifier, train_loader, valid_loader, epochs=100)

    # save the model
    torch.save(mlp_classifier.state_dict(), 'mlp_classifier.pth')




    #----- PLOT LOGS--------

    plot_logs_classification(mlp_classifier_logs)

    # save plot
    plt.savefig('loss_mlp_classifier', format="jpg", bbox_inches='tight', pad_inches=0)




    # ----- ACCURACY --------

    mlp_classifier_predictions,mlp_classifier_gt = test_model_classification(mlp_classifier, valid_loader)

    #print "Accuracy MLP Classifier: %0.2f" % \
        #accuracy_score(mlp_classifier_gt,mlp_classifier_predictions.argmax(1))

    # save on txt file
    with open("acc_test_mlp_classifier.txt", "w") as text_file:
        text_file.write("Accuracy MLP Classifier: {:.2f}".format(accuracy_score(mlp_classifier_gt,mlp_classifier_predictions.argmax(1))))





    # ---- CONFUSION MATRIX ----

    conf_matrix =  confusion_matrix(mlp_classifier_gt, mlp_classifier_predictions.argmax(1))
    class_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]



    # Plot non-normalized confusion matrix
    plt.figure(figsize=(20,20))
    plt.subplot(211)
    plot_confusion_matrix(conf_matrix, classes=class_names,
                        title='Confusion matrix MLP')


    # Plot normalized confusion matrix
    plt.subplot(212)
    plot_confusion_matrix(conf_matrix, classes=class_names, normalize=True,
                        title='Normalized Confusion Matrix MLP')


    #plot.show()

    # save plot
    plt.savefig('conf_matrix_mlp_classifier', format="jpg", bbox_inches='tight', pad_inches=0)







    # ---- SCORE F1 AND mF1----

    scores =  f1_score(mlp_classifier_gt,mlp_classifier_predictions.argmax(1), average=None)
    #print scores

    #print scores.mean() #mF1

    # save on txt file
    with open("score_F1_mF1_mlp_classifier.txt", "w") as text_file:
        text_file.write("Score F1 Classe0 MLP Classifier: {:.2f}\nScore F1 Classe1 MLP Classifier: {:.2f}\
        \nScore F1 Classe2 MLP Classifier: {:.2f}\nScore F1 Classe3 MLP Classifier: {:.2f}\
        \nScore F1 Classe4 MLP Classifier: {:.2f}\nScore F1 Classe5 MLP Classifier: {:.2f}\
        \nScore F1 Classe6 MLP Classifier: {:.2f}\nScore F1 Classe7 MLP Classifier: {:.2f}\
        \nScore F1 Classe8 MLP Classifier: {:.2f}\nScore F1 Classe9 MLP Classifier: {:.2f}\
        \nScore F1 Classe10 MLP Classifier: {:.2f}\nScore F1 Classe11 MLP Classifier: {:.2f}\
        \nScore F1 Classe12 MLP Classifier: {:.2f}\nScore F1 Classe13 MLP Classifier: {:.2f}\
        \nScore F1 Classe14 MLP Classifier: {:.2f}\nScore F1 Classe15 MLP Classifier: {:.2f}\
        \nScore mF1 MLP Classifier: {:.2f}".format(scores[0], scores[1], scores[2], scores[3], scores[4],
                                                scores[5], scores[6], scores[7], scores[8], scores[9],
                                                scores[10], scores[11], scores[12], scores[13], scores[14],
                                                scores[15], scores.mean()))






def deep_mlp():

    # ------ MODEL -------
    deep_mlp_classifier = DeepMLPClassifier(110592, 16, 512) # input, classi, hidden neurons of 2 levels


    # training set
    train = Dataset('dataset/images','dataset/training_list.csv',transform=transforms.ToTensor())




    #------ NORMALIZATION ---------

    m,s = normalization(train)




    # --------- DATALOADER AND TRANSFORMATION ----------

    transform = transforms.Compose([transforms.ToTensor(), #conversione in tensore
                                    transforms.Normalize(m,s), # nomrlizzazione con media e dvst del TS
                                    transforms.Lambda(lambda x: x.view(-1))]) # trasforma l'immagine in un unico vettore

    #ridefiniamo il training set specificando le trasformazioni
    train = Dataset('dataset/images','dataset/training_list.csv',transform=transform)

    # permutiamo i dati di training
    idx = np.random.permutation(len(train))
    train.images = train.images[idx]

    #print "Immagine di train:", train[0]['image'].shape # 3x144x256
    #print "Etichetta:", train[0]['label']
    #print "Posa:", train[0]['pose']



    #print ""


    #ridefiniamo il test set specificando le trasformazioni
    valid = Dataset('dataset/images','dataset/validation_list.csv',transform=transform)

    # permutiamo i dati di validation
    idx = np.random.permutation(len(valid))
    valid.images = valid.images[idx]

    #print "Immagine di validation:", valid[0]['image'].shape # 3x144x256
    #print "Etichetta:", valid[0]['label']
    #print "Posa:", valid[0]['pose']


    #definiamo i dataloaders
    train_loader = DataLoader(train, batch_size=64, num_workers=2, shuffle=True) #shuffle accede ai dati in maniera casuale
    #shuffle permette di accedere ai dati in maniera casuale
    valid_loader = DataLoader(valid, batch_size=64, num_workers=2)





    #----- START TRAINING -------

    deep_mlp_classifier, deep_mlp_classifier_logs = train_classification(deep_mlp_classifier, train_loader, valid_loader, epochs=100)

    # save the model
    torch.save(deep_mlp_classifier.state_dict(), 'deep_mlp_classifier.pth')




    #----- PLOT LOGS--------

    plot_logs_classification(deep_mlp_classifier_logs)

    # save plot
    plt.savefig('loss_deep_mlp_classifier', format="jpg", bbox_inches='tight', pad_inches=0)




    # ----- ACCURACY --------

    deep_mlp_classifier_predictions,deep_mlp_classifier_gt = test_model_classification(deep_mlp_classifier, valid_loader)

    #print "Accuracy DEEP MLP Classifier: %0.2f" % \
        #accuracy_score(deep_mlp_classifier_gt,deep_mlp_classifier_predictions.argmax(1))

    # save on txt file
    with open("acc_test_deep_mlp_classifier.txt", "w") as text_file:
        text_file.write("AccuracyDEEP MLP Classifier: {:.2f}".format(accuracy_score(deep_mlp_classifier_gt,deep_mlp_classifier_predictions.argmax(1))))





    # ---- CONFUSION MATRIX ----

    conf_matrix =  confusion_matrix(deep_mlp_classifier_gt, deep_mlp_classifier_predictions.argmax(1))
    class_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]



    # Plot non-normalized confusion matrix
    plt.figure(figsize=(20,20))
    plt.subplot(211)
    plot_confusion_matrix(conf_matrix, classes=class_names,
                      title='Confusion matrix DEEP MLP')


    # Plot normalized confusion matrix
    plt.subplot(212)
    plot_confusion_matrix(conf_matrix, classes=class_names, normalize=True,
                      title='Normalized Confusion Matrix DEEP MLP')


    #plot.show()

    # save plot
    plt.savefig('conf_matrix_deep_mlp_classifier', format="jpg", bbox_inches='tight', pad_inches=0)







    # ---- SCORE F1 AND mF1----

    scores =  f1_score(deep_mlp_classifier_gt,deep_mlp_classifier_predictions.argmax(1), average=None)
    #print scores

    #print scores.mean() #mF1

    # save on txt file
    with open("score_F1_mF1_deep_mlp_classifier.txt", "w") as text_file:
        text_file.write("Score F1 Classe0 DEEP MLP Classifier: {:.2f}\nScore F1 Classe1 DEEP MLP Classifier: {:.2f}\
        \nScore F1 Classe2 DEEP MLP Classifier: {:.2f}\nScore F1 Classe3 DEEP MLP Classifier: {:.2f}\
        \nScore F1 Classe4 DEEP MLP Classifier: {:.2f}\nScore F1 Classe5 DEEP MLP Classifier: {:.2f}\
        \nScore F1 Classe6 DEEP MLP Classifier: {:.2f}\nScore F1 Classe7 DEEP MLP Classifier: {:.2f}\
        \nScore F1 Classe8 DEEP MLP Classifier: {:.2f}\nScore F1 Classe9 DEEP MLP Classifier: {:.2f}\
        \nScore F1 Classe10 DEEP MLP Classifier: {:.2f}\nScore F1 Classe11 DEEP MLP Classifier: {:.2f}\
        \nScore F1 Classe12 DEEP MLP Classifier: {:.2f}\nScore F1 Classe13 DEEP MLP Classifier: {:.2f}\
        \nScore F1 Classe14 DEEP MLP Classifier: {:.2f}\nScore F1 Classe15 DEEP MLP Classifier: {:.2f}\
        \nScore mF1 DEEP MLP Classifier: {:.2f}".format(scores[0], scores[1], scores[2], scores[3], scores[4],
                                                scores[5], scores[6], scores[7], scores[8], scores[9],
                                                scores[10], scores[11], scores[12], scores[13], scores[14],
                                                scores[15], scores.mean()))



#-----------------------------------------------------------------------------------------------------