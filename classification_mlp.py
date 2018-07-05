#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
from classification_models import MLPClassifier
from classification import *
from dataset import Dataset




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

