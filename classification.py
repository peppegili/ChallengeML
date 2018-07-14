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
from torch.utils.data import DataLoader
from dataset import Dataset
from classification_models import *
from torchvision import models, transforms
from copy import deepcopy



#-----------------------------------------------------------------------------------------

# ------- TRAINING PROCEDURE -------

def train_classification(model, train_loader, test_loader, lr=0.00001, epochs=100, momentum=0.9, weight_decay = 0.000001):
    since = time.time()
    print "---- TRAINING START ----"

    best_model = copy.deepcopy(model.state_dict())  # for best model
    best_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=momentum, weight_decay=weight_decay)

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
    train_loader = DataLoader(train, batch_size=32, num_workers=2, shuffle=True) #shuffle accede ai dati in maniera casuale
    #shuffle permette di accedere ai dati in maniera casuale
    valid_loader = DataLoader(valid, batch_size=32, num_workers=2)





    #----- START TRAINING -------

    mlp_classifier, mlp_classifier_logs = train_classification(mlp_classifier, train_loader, valid_loader, epochs=150)

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
    train_loader = DataLoader(train, batch_size=32, num_workers=2, shuffle=True) #shuffle accede ai dati in maniera casuale
    #shuffle permette di accedere ai dati in maniera casuale
    valid_loader = DataLoader(valid, batch_size=32, num_workers=2)





    #----- START TRAINING -------

    deep_mlp_classifier, deep_mlp_classifier_logs = train_classification(deep_mlp_classifier, train_loader, valid_loader, epochs=150)

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










# architettura VGG16 con fine tuning del modulo "classifier"(i fully connected layers finali)
def vgg16_fc():

    #-------MODEL--------
    vgg16_orig = models.vgg16(pretrained=True)  # utilizziamo i pesi già allenati

    # facciamo una copia del modello
    vgg16 = deepcopy(vgg16_orig)
    # vgg16 = vgg16.cuda()

    for param in vgg16.features.parameters():
        param.requires_grad = False  # freeze dei layer convoluzionali


    features = list(vgg16.classifier.children())[1:-1]  # rimozione primo e ultimo layer

    del features[2]  # rimozione quarto livello

    features.insert(0, nn.Linear(16384, 3072))  # aggiungiamo il primo layer # img 144x256
    features.insert(3, nn.Linear(3072, 3072))  # aggiungiamo il quarto layer
    features.append(nn.Linear(3072, 16))  # aggiungiamo layer con 16 output

    vgg16.classifier = nn.Sequential(*features)  # sostituiamo il modulo "classifier"

    #print vgg16




    # --------- DATALOADER AND TRANSFORMATION ----------

    transform = transforms.Compose([transforms.ToTensor(),  # conversione in tensore
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # normalizzazione con media e dvst di vgg16

    # ridefiniamo il training set specificando le trasformazioni
    train = Dataset('dataset/images', 'dataset/training_list.csv', transform=transform)

    # permutiamo i dati di training
    idx = np.random.permutation(len(train))
    train.images = train.images[idx]

    # print "Immagine di train:", train[0]['image'].shape # 3x144x256
    # print "Etichetta:", train[0]['label']
    # print "Posa:", train[0]['pose']

    # print ""

    # ridefiniamo il test set specificando le trasformazioni
    valid = Dataset('dataset/images', 'dataset/validation_list.csv', transform=transform)

    # permutiamo i dati di validation
    idx = np.random.permutation(len(valid))
    valid.images = valid.images[idx]

    # print "Immagine di validation:", valid[0]['image'].shape # 3x144x256
    # print "Etichetta:", valid[0]['label']
    # print "Posa:", valid[0]['pose']

    # definiamo i dataloaders
    train_loader = DataLoader(train, batch_size=32, num_workers=2, shuffle=True)  # shuffle accede ai dati in maniera casuale
    valid_loader = DataLoader(valid, batch_size=32, num_workers=2)



    #-------- START TRAINING ----------

    vgg16_fc_classifier, vgg16_fc_classifier_logs = train_classification(vgg16, train_loader, valid_loader, epochs=400)

    # save the model
    torch.save(vgg16_fc_classifier.state_dict(), 'vgg16_fc_classifier.pth')




    # ----- PLOT LOGS--------

    plot_logs_classification(vgg16_fc_classifier_logs)

    # save plot
    plt.savefig('loss_vgg16_fc_classifier', format="jpg", bbox_inches='tight', pad_inches=0)




    # ----- ACCURACY --------

    vgg16_fc_classifier_predictions, vgg16_fc_classifier_gt = test_model_classification(vgg16_fc_classifier, valid_loader)

    # print "Accuracy VGG16 FC Classifier: %0.2f" % \
        # accuracy_score(vgg16_fc_classifier_gt,vgg16_fc_classifier_predictions.argmax(1))

    # save on txt file
    with open("acc_test_vgg16_fc_classifier.txt", "w") as text_file:
        text_file.write("Accuracy VGG16 FC Classifier: {:.2f}".format(accuracy_score(vgg16_fc_classifier_gt, vgg16_fc_classifier_predictions.argmax(1))))




    # ---- CONFUSION MATRIX ----

    conf_matrix_vgg16_fc = confusion_matrix(vgg16_fc_classifier_gt, vgg16_fc_classifier_predictions.argmax(1))
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(20, 20))
    plt.subplot(211)
    plot_confusion_matrix(conf_matrix_vgg16_fc, classes=class_names,
                          title='Confusion matrix VGG16 FC fine-tuning')

    # Plot normalized confusion matrix
    plt.subplot(212)
    plot_confusion_matrix(conf_matrix_vgg16_fc, classes=class_names, normalize=True,
                          title='Normalized Confusion Matrix VGG16 FC fine-tuning')

    # plot.show()

    # save plot
    plt.savefig('conf_matrix_vgg16_fc_classifier', format="jpg", bbox_inches='tight', pad_inches=0)





    # ---- SCORE F1 AND mF1----

    scores_vgg16_fc = f1_score(vgg16_fc_classifier_gt, vgg16_fc_classifier_predictions.argmax(1), average=None)
    # print "Score F1 VGG16 FC:", scores_vgg16_fc

    # print "Score mF1 VGG16 FC:", scores_vgg16_fc.mean() #mF1

    # save on txt file
    with open("score_F1_mF1_vgg16_fc_classifier.txt", "w") as text_file:
        text_file.write("Score F1 Classe0 VGG16 FC Classifier: {:.2f}\nScore F1 Classe1 VGG16 FC Classifier: {:.2f}\
        \nScore F1 Classe2 VGG16 FC Classifier: {:.2f}\nScore F1 Classe3 VGG16 FC Classifier: {:.2f}\
        \nScore F1 Classe4 VGG16 FC Classifier: {:.2f}\nScore F1 Classe5 VGG16 FC Classifier: {:.2f}\
        \nScore F1 Classe6 VGG16 FC Classifier: {:.2f}\nScore F1 Classe7 VGG16 FC Classifier: {:.2f}\
        \nScore F1 Classe8 VGG16 FC Classifier: {:.2f}\nScore F1 Classe9 VGG16 FC Classifier: {:.2f}\
        \nScore F1 Classe10 VGG16 FC Classifier: {:.2f}\nScore F1 Classe11 VGG16 FC Classifier: {:.2f}\
        \nScore F1 Classe12 VGG16 FC Classifier: {:.2f}\nScore F1 Classe13 VGG16 FC Classifier: {:.2f}\
        \nScore F1 Classe14 VGG16 FC Classifier: {:.2f}\nScore F1 Classe15 VGG16 FC Classifier: {:.2f}\
        \nScore mF1 VGG16 FC Classifier: {:.2f}".format(scores_vgg16_fc[0], scores_vgg16_fc[1], scores_vgg16_fc[2],
                                                        scores_vgg16_fc[3], scores_vgg16_fc[4],
                                                        scores_vgg16_fc[5], scores_vgg16_fc[6], scores_vgg16_fc[7],
                                                        scores_vgg16_fc[8], scores_vgg16_fc[9],
                                                        scores_vgg16_fc[10], scores_vgg16_fc[11], scores_vgg16_fc[12],
                                                        scores_vgg16_fc[13], scores_vgg16_fc[14],
                                                        scores_vgg16_fc[15], scores_vgg16_fc.mean()))





# architettura VGG16 con fine tuning del modulo "classifier"(i fully connected layers finali)
# e dell'ultimo blocco convoluzionale del modulo "features"
def vgg16_cl_fc():

    #----MODEL-----
    vgg16_orig = models.vgg16(pretrained=True)  # utilizziamo i pesi già allenati

    # facciamo una copia del modello
    vgg16 = deepcopy(vgg16_orig)
    # vgg16 = vgg16.cuda()

    for param in vgg16.features.parameters():
        param.requires_grad = False  # freeze dei layer convoluzionali

    #vgg16_trainable_parameters = filter(lambda p: p.requires_grad, vgg16.parameters())
    #print "Numero di parametri trainabili vgg16: ", sum([p.numel() for p in vgg16_trainable_parameters])

    # sfreez dell'ultimo blocco convoluzionale(dal livello 24 al 30) del modulo "features"
    layer = -1
    for child in vgg16.features.children():
        # print child
        layer += 1
        if layer > 23 and layer < 31: # provato anche con layer>16
            #print"Sfreezato -> ", child
            for param in child.parameters():
                param.requires_grad = True

    #vgg16_trainable_parameters = filter(lambda p: p.requires_grad, vgg16.parameters())
    #print "Numero di parametri trainabili vgg16: ", sum([p.numel() for p in vgg16_trainable_parameters])

    features = list(vgg16.classifier.children())[1:-1]  # rimozione primo e ultimo layer

    del features[2]  # rimozione quarto livello

    features.insert(0, nn.Linear(16384, 3072))  # aggiungiamo il primo layer # img 144x256
    features.insert(3, nn.Linear(3072, 3072))  # aggiungiamo il quarto layer
    features.append(nn.Linear(3072, 16))  # aggiungiamo layer con 16 output

    vgg16.classifier = nn.Sequential(*features)  # sostituiamo il modulo "classifier"

    #print vgg16




    # --------- DATALOADER AND TRANSFORMATION ----------

    transform = transforms.Compose([transforms.ToTensor(),  # conversione in tensore
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # normalizzazione con media e dvst di vgg16

    # ridefiniamo il training set specificando le trasformazioni
    train = Dataset('dataset/images', 'dataset/training_list.csv', transform=transform)

    # permutiamo i dati di training
    idx = np.random.permutation(len(train))
    train.images = train.images[idx]

    # print "Immagine di train:", train[0]['image'].shape # 3x144x256
    # print "Etichetta:", train[0]['label']
    # print "Posa:", train[0]['pose']

    # print ""

    # ridefiniamo il test set specificando le trasformazioni
    valid = Dataset('dataset/images', 'dataset/validation_list.csv', transform=transform)

    # permutiamo i dati di validation
    idx = np.random.permutation(len(valid))
    valid.images = valid.images[idx]

    # print "Immagine di validation:", valid[0]['image'].shape # 3x144x256
    # print "Etichetta:", valid[0]['label']
    # print "Posa:", valid[0]['pose']

    # definiamo i dataloaders
    train_loader = DataLoader(train, batch_size=32, num_workers=2, shuffle=True)  # shuffle accede ai dati in maniera casuale
    valid_loader = DataLoader(valid, batch_size=32, num_workers=2)



    # -------- START TRAINING ----------

    vgg16_cl_fc_classifier, vgg16_cl_fc_classifier_logs = train_classification(vgg16, train_loader, valid_loader, epochs=400)

    # save the model
    torch.save(vgg16_cl_fc_classifier.state_dict(), 'vgg16_cl_fc_classifier.pth')



    # ----- PLOT LOGS--------

    plot_logs_classification(vgg16_cl_fc_classifier_logs)

    # save plot
    plt.savefig('loss_vgg16_cl_fc_classifier', format="jpg", bbox_inches='tight', pad_inches=0)



    # ----- ACCURACY --------

    vgg16_cl_fc_classifier_predictions, vgg16_cl_fc_classifier_gt = test_model_classification(vgg16_cl_fc_classifier, valid_loader)

    # print "Accuracy VGG16 CL FC Classifier: %0.2f" % \
        # accuracy_score(vgg16_cl_fc_classifier_gt,vgg16_cl_fc_classifier_predictions.argmax(1))

    # save on txt file
    with open("acc_test_vgg16_cl_fc_classifier.txt", "w") as text_file:
        text_file.write("Accuracy VGG16 CL FC Classifier: {:.2f}".format(accuracy_score(vgg16_cl_fc_classifier_gt, vgg16_cl_fc_classifier_predictions.argmax(1))))




    # ---- CONFUSION MATRIX ----

    conf_matrix_vgg16_cl_fc = confusion_matrix(vgg16_cl_fc_classifier_gt, vgg16_cl_fc_classifier_predictions.argmax(1))
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(20, 20))
    plt.subplot(211)
    plot_confusion_matrix(conf_matrix_vgg16_cl_fc, classes=class_names,
                          title='Confusion matrix VGG16 CL FC fine-tuning')

    # Plot normalized confusion matrix
    plt.subplot(212)
    plot_confusion_matrix(conf_matrix_vgg16_cl_fc, classes=class_names, normalize=True,
                          title='Normalized Confusion Matrix VGG16 CL FC fine-tuning')

    # plot.show()

    # save plot
    plt.savefig('conf_matrix_vgg16_cl_fc_classifier', format="jpg", bbox_inches='tight', pad_inches=0)



    # ---- SCORE F1 AND mF1----

    scores_vgg16_cl_fc = f1_score(vgg16_cl_fc_classifier_gt, vgg16_cl_fc_classifier_predictions.argmax(1), average=None)
    #print "Score F1 VGG16 CL FC:", scores_vgg16_cl_fc

    #print "Score mF1 VGG16 CL FC:", scores_vgg16_cl_fc.mean()  # mF1

    # save on txt file
    with open("score_F1_mF1_vgg16_cl_fc_classifier.txt", "w") as text_file:
        text_file.write("Score F1 Classe0 VGG16 CL FC Classifier: {:.2f}\nScore F1 Classe1 VGG16 CL FC Classifier: {:.2f}\
        \nScore F1 Classe2 VGG16 CL FC Classifier: {:.2f}\nScore F1 Classe3 VGG16 CL FC Classifier: {:.2f}\
        \nScore F1 Classe4 VGG16 CL FC Classifier: {:.2f}\nScore F1 Classe5 VGG16 CL FC Classifier: {:.2f}\
        \nScore F1 Classe6 VGG16 CL FC Classifier: {:.2f}\nScore F1 Classe7 VGG16 CL FC Classifier: {:.2f}\
        \nScore F1 Classe8 VGG16 CL FC Classifier: {:.2f}\nScore F1 Classe9 VGG16 CL FC Classifier: {:.2f}\
        \nScore F1 Classe10 VGG16 CL FC Classifier: {:.2f}\nScore F1 Classe11 VGG16 CL FC Classifier: {:.2f}\
        \nScore F1 Classe12 VGG16 CL FC Classifier: {:.2f}\nScore F1 Classe13 VGG16 CL FC Classifier: {:.2f}\
        \nScore F1 Classe14 VGG16 CL FC Classifier: {:.2f}\nScore F1 Classe15 VGG16 CL FC Classifier: {:.2f}\
        \nScore mF1 VGG16 CL FC Classifier: {:.2f}".format(scores_vgg16_cl_fc[0], scores_vgg16_cl_fc[1],
                                                           scores_vgg16_cl_fc[2], scores_vgg16_cl_fc[3],
                                                           scores_vgg16_cl_fc[4], scores_vgg16_cl_fc[5],
                                                           scores_vgg16_cl_fc[6], scores_vgg16_cl_fc[7],
                                                           scores_vgg16_cl_fc[8], scores_vgg16_cl_fc[9],
                                                           scores_vgg16_cl_fc[10], scores_vgg16_cl_fc[11],
                                                           scores_vgg16_cl_fc[12], scores_vgg16_cl_fc[13],
                                                           scores_vgg16_cl_fc[14], scores_vgg16_cl_fc[15],
                                                           scores_vgg16_cl_fc.mean()))









# architettura VGG16 con fine tuning del modulo "classifier"(i fully connected layers finali)
# e degli ultimi due blocchi convoluzionali del modulo "features". Inoltre, si effettua la data augmentation
def vgg16_cl_fc_aug():

    #----MODEL-----
    vgg16_orig = models.vgg16(pretrained=True)  # utilizziamo i pesi già allenati

    # facciamo una copia del modello
    vgg16 = deepcopy(vgg16_orig)
    # vgg16 = vgg16.cuda()

    for param in vgg16.features.parameters():
        param.requires_grad = False  # freeze dei layer convoluzionali

    #vgg16_trainable_parameters = filter(lambda p: p.requires_grad, vgg16.parameters())
    #print "Numero di parametri trainabili vgg16: ", sum([p.numel() for p in vgg16_trainable_parameters])

    # sfreez degli ultimi due blocchi convoluzionali(dal livello 17 al 30) del modulo "features"
    layer = -1
    for child in vgg16.features.children():
        # print child
        layer += 1
        if layer > 16 and layer < 31:
            #print"Sfreezato -> ", child
            for param in child.parameters():
                param.requires_grad = True


    # cambiamo i dropout nel blocco fully connected
    for layer in vgg16.classifier.children():
        if (type(layer) == nn.Dropout):
            layer.p = 0.25

    #vgg16_trainable_parameters = filter(lambda p: p.requires_grad, vgg16.parameters())
    #print "Numero di parametri trainabili vgg16: ", sum([p.numel() for p in vgg16_trainable_parameters])

    features = list(vgg16.classifier.children())[1:-1]  # rimozione primo e ultimo layer

    del features[2]  # rimozione quarto livello

    features.insert(0, nn.Linear(8192, 2048))  # aggiungiamo il primo layer # img 144x256
    features.insert(3, nn.Linear(2048, 2048))  # aggiungiamo il quarto layer
    features.append(nn.Linear(2048, 16))  # aggiungiamo layer con 16 output

    vgg16.classifier = nn.Sequential(*features)  # sostituiamo il modulo "classifier"

    #print vgg16




    # --------- DATALOADER AND TRANSFORMATION ----------

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(),
                                          transforms.RandomRotation(20), # random rotate between -20 +20 degree
                                          transforms.RandomCrop(128), # square crop 128x128
                                          transforms.ToTensor(),  # conversione in tensore
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # normalizzazione con media e dvst di vgg16

    # ridefiniamo il training set specificando le trasformazioni
    train = Dataset('dataset/images', 'dataset/training_list.csv', transform=transform_train)

    # permutiamo i dati di training
    idx = np.random.permutation(len(train))
    train.images = train.images[idx]

    # print "Immagine di train:", train[0]['image'].shape # 3x144x256
    # print "Etichetta:", train[0]['label']
    # print "Posa:", train[0]['pose']

    # print ""

    transform_valid = transforms.Compose([transforms.CenterCrop(128),  # square crop 128x128
                                          transforms.ToTensor(),  # conversione in tensore
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # ridefiniamo il test set specificando le trasformazioni
    valid = Dataset('dataset/images', 'dataset/validation_list.csv', transform=transform_valid)

    # permutiamo i dati di validation
    idx = np.random.permutation(len(valid))
    valid.images = valid.images[idx]

    # print "Immagine di validation:", valid[0]['image'].shape # 3x144x256
    # print "Etichetta:", valid[0]['label']
    # print "Posa:", valid[0]['pose']

    # definiamo i dataloaders
    train_loader = DataLoader(train, batch_size=32, num_workers=2, shuffle=True)  # shuffle accede ai dati in maniera casuale
    valid_loader = DataLoader(valid, batch_size=32, num_workers=2)



    # -------- START TRAINING ----------

    vgg16_cl_fc_aug_classifier, vgg16_cl_fc_aug_classifier_logs = train_classification(vgg16, train_loader, valid_loader, epochs=400)

    # save the model
    torch.save(vgg16_cl_fc_aug_classifier.state_dict(), 'vgg16_cl_fc_aug_classifier.pth')



    # ----- PLOT LOGS--------

    plot_logs_classification(vgg16_cl_fc_aug_classifier_logs)

    # save plot
    plt.savefig('loss_vgg16_cl_fc_aug_classifier', format="jpg", bbox_inches='tight', pad_inches=0)



    # ----- ACCURACY --------

    vgg16_cl_fc_aug_classifier_predictions, vgg16_cl_fc_aug_classifier_gt = test_model_classification(vgg16_cl_fc_aug_classifier, valid_loader)

    # print "Accuracy VGG16 CL FC Classifier: %0.2f" % \
        # accuracy_score(vgg16_cl_fc_classifier_gt,vgg16_cl_fc_classifier_predictions.argmax(1))

    # save on txt file
    with open("acc_test_vgg16_cl_fc_aug_classifier.txt", "w") as text_file:
        text_file.write("Accuracy VGG16 CL FC AUG Classifier: {:.2f}".format(accuracy_score(vgg16_cl_fc_aug_classifier_gt, vgg16_cl_fc_aug_classifier_predictions.argmax(1))))




    # ---- CONFUSION MATRIX ----

    conf_matrix_vgg16_cl_fc_aug = confusion_matrix(vgg16_cl_fc_aug_classifier_gt, vgg16_cl_fc_aug_classifier_predictions.argmax(1))
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(20, 20))
    plt.subplot(211)
    plot_confusion_matrix(conf_matrix_vgg16_cl_fc_aug, classes=class_names,
                          title='Confusion matrix VGG16 CL FC AUG fine-tuning')

    # Plot normalized confusion matrix
    plt.subplot(212)
    plot_confusion_matrix(conf_matrix_vgg16_cl_fc_aug, classes=class_names, normalize=True,
                          title='Normalized Confusion Matrix VGG16 CL FC AUG fine-tuning')

    # plot.show()

    # save plot
    plt.savefig('conf_matrix_vgg16_cl_fc_aug_classifier', format="jpg", bbox_inches='tight', pad_inches=0)



    # ---- SCORE F1 AND mF1----

    scores_vgg16_cl_fc_aug = f1_score(vgg16_cl_fc_aug_classifier_gt, vgg16_cl_fc_aug_classifier_predictions.argmax(1), average=None)
    #print "Score F1 VGG16 CL FC AUG:", scores_vgg16_cl_fc_aug

    #print "Score mF1 VGG16 CL FC AUG:", scores_vgg16_cl_fc_aug.mean()  # mF1

    # save on txt file
    with open("score_F1_mF1_vgg16_cl_fc_aug_classifier.txt", "w") as text_file:
        text_file.write("Score F1 Classe0 VGG16 CL FC AUG Classifier: {:.2f}\nScore F1 Classe1 VGG16 CL FC AUG Classifier: {:.2f}\
        \nScore F1 Classe2 VGG16 CL FC AUG Classifier: {:.2f}\nScore F1 Classe3 VGG16 CL FC AUG Classifier: {:.2f}\
        \nScore F1 Classe4 VGG16 CL FC AUG Classifier: {:.2f}\nScore F1 Classe5 VGG16 CL FC AUG Classifier: {:.2f}\
        \nScore F1 Classe6 VGG16 CL FC AUG Classifier: {:.2f}\nScore F1 Classe7 VGG16 CL FC AUG Classifier: {:.2f}\
        \nScore F1 Classe8 VGG16 CL FC AUG Classifier: {:.2f}\nScore F1 Classe9 VGG16 CL FC AUG Classifier: {:.2f}\
        \nScore F1 Classe10 VGG16 CL FC AUG Classifier: {:.2f}\nScore F1 Classe11 VGG16 CL FC AUG Classifier: {:.2f}\
        \nScore F1 Classe12 VGG16 CL FC AUG Classifier: {:.2f}\nScore F1 Classe13 VGG16 CL FC AUG Classifier: {:.2f}\
        \nScore F1 Classe14 VGG16 CL FC AUG Classifier: {:.2f}\nScore F1 Classe15 VGG16 CL FC AUG Classifier: {:.2f}\
        \nScore mF1 VGG16 CL FC AUG Classifier: {:.2f}".format(scores_vgg16_cl_fc_aug[0], scores_vgg16_cl_fc_aug[1],
                                                           scores_vgg16_cl_fc_aug[2], scores_vgg16_cl_fc_aug[3],
                                                           scores_vgg16_cl_fc_aug[4], scores_vgg16_cl_fc_aug[5],
                                                           scores_vgg16_cl_fc_aug[6], scores_vgg16_cl_fc_aug[7],
                                                           scores_vgg16_cl_fc_aug[8], scores_vgg16_cl_fc_aug[9],
                                                           scores_vgg16_cl_fc_aug[10], scores_vgg16_cl_fc_aug[11],
                                                           scores_vgg16_cl_fc_aug[12], scores_vgg16_cl_fc_aug[13],
                                                           scores_vgg16_cl_fc_aug[14], scores_vgg16_cl_fc_aug[15],
                                                           scores_vgg16_cl_fc_aug.mean()))



#-----------------------------------------------------------------------------------------------------