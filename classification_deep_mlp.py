#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
from classification_models import DeepMLPClassifier
from classification import *
from dataset import Dataset



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
