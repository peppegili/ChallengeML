#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.optim import SGD
from torch.autograd import Variable
import time
import copy
import numpy as np
import csv
from matplotlib import pyplot as plt
from regression_models import *
from dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn import metrics
from evaluate import main, evaluate_localization




#-----------------------------------------------------------------------------------------


# ------- TRAINING PROCEDURE -------

def train_regression(model, train_loader, test_loader, lr=0.00001, epochs=100, momentum=0.9):
    since = time.time()
    print "---- TRAINING START ----"

    # best_model = copy.deepcopy(model.state_dict()) # for best model
    # best_tot_loss = 0.0

    criterion = nn.MSELoss()  # Loss MSE per allenare il regressore
    optimizer = SGD(model.parameters(), lr, momentum=momentum)  # ottimizzatore

    losses1 = {'train': [], 'test': []}  # dizionario che conterrà le liste delle loss di training e testing
    losses2 = {'train': [], 'test': []}
    losses3 = {'train': [], 'test': []}
    losses4 = {'train': [], 'test': []}
    loaders = {'train': train_loader, 'test': test_loader}  # dizionario contenente i dataloaders

    if torch.cuda.is_available():
        model = model.cuda()

    for e in range(epochs):
        for mode in ['train', 'test']:
            if mode == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss1 = 0
            epoch_loss2 = 0
            epoch_loss3 = 0
            epoch_loss4 = 0
            samples = 0

            for i, batch in enumerate(loaders[mode]):  # utilizziamo il data loader appropriato
                # trasformiamo i tensori in variabili
                x = Variable(batch['image'], requires_grad=(mode == 'train'))  # i gradienti ci servono solo in fase di training
                y1 = Variable(batch['pose'][0]).float()  # x
                y2 = Variable(batch['pose'][1]).float()  # y
                y3 = Variable(batch['pose'][2]).float()  # u
                y4 = Variable(batch['pose'][3]).float()  # v

                if torch.cuda.is_available():
                    x, y1, y2, y3, y4 = x.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()

                output = model(x)
                l1 = criterion(output[0], y1)
                l2 = criterion(output[1], y2)
                l3 = criterion(output[2], y3)
                l4 = criterion(output[3], y4)
                l = l1 + l2 + l3 + l4

                if mode == 'train':
                    l.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss1 += l1.data[0] * x.shape[0]
                epoch_loss2 += l2.data[0] * x.shape[0]
                epoch_loss3 += l3.data[0] * x.shape[0]
                epoch_loss4 += l4.data[0] * x.shape[0]
                samples += x.shape[0]

                print "\r[%s] Epoch %d/%d. Iteration %d/%d. Loss1: %0.2f. Loss2: %0.2f. Loss3: %0.2f. Loss4: %0.2f." % \
                      (mode, e + 1, epochs, i, len(loaders[mode]), epoch_loss1 / samples, epoch_loss2 / samples,
                       epoch_loss3 / samples, epoch_loss4 / samples),

            epoch_loss1 /= samples
            epoch_loss2 /= samples
            epoch_loss3 /= samples
            epoch_loss4 /= samples

            losses1[mode].append(epoch_loss1)
            losses2[mode].append(epoch_loss2)
            losses3[mode].append(epoch_loss3)
            losses4[mode].append(epoch_loss4)

            print "\r[%s] Epoch %d/%d. Iteration %d/%d. Loss1: %0.2f. Loss2: %0.2f. Loss3: %0.2f. Loss4: %0.2f." % \
                  (mode, e + 1, epochs, i, len(loaders[mode]), epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4)

            # deep copy the best model
            # if mode == 'test' and (epoch_loss1+epoch_loss2+epoch_loss3+epoch_loss4) < best_tot_loss:
            # best_tot_loss = epoch_loss1+epoch_loss2+epoch_loss3+epoch_loss4
            # best_model = copy.deepcopy(model.state_dict())

    print""

    time_elapsed = time.time() - since
    print('---- TRAINING COMPLETE ---- in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    # model.load_state_dict(best_model)
    # torch.save(model.state_dict(),'regression_model.pth')

    # restituiamo il modello e i log delle loss
    return model, (losses1, losses2, losses3, losses4)








# ----- PLOT LOGS--------

def plot_logs_regression(logs):
    train_losses1, test_losses1 = logs[0]['train'], logs[0]['test']
    train_losses2, test_losses2 = logs[1]['train'], logs[1]['test']
    train_losses3, test_losses3 = logs[2]['train'], logs[2]['test']
    train_losses4, test_losses4 = logs[3]['train'], logs[3]['test']

    plt.figure(figsize=(18, 10))
    plt.subplot(221)
    plt.title('Regression Loss1 (x)')
    plt.plot(train_losses1)
    plt.plot(test_losses1)
    plt.legend(['Training Loss', 'Test Losses'])
    plt.grid()

    plt.subplot(222)
    plt.title('Regression Loss2 (y)')
    plt.plot(train_losses2)
    plt.plot(test_losses2)
    plt.legend(['Training Loss', 'Test Losses'])
    plt.grid()

    plt.subplot(223)
    plt.title('Regression Loss3 (u)')
    plt.plot(train_losses3)
    plt.plot(test_losses3)
    plt.legend(['Training Loss', 'Test Losses'])
    plt.grid()

    plt.subplot(224)
    plt.title('Regression Loss4 (v)')
    plt.plot(train_losses4)
    plt.plot(test_losses4)
    plt.legend(['Training Loss', 'Test Losses'])
    plt.grid()

    # plt.show()







# ----- TEST MODEL -------

def test_model_regression(model, test_loader, predicted_pose_str, target_pose_str):
    model.eval()
    preds1 = []  # conterra la posa x stimata
    gts1 = []
    preds2 = []  # conterra la posa y stimata
    gts2 = []
    preds3 = []  # conterra la posa u stimata
    gts3 = []
    preds4 = []  # conterra la posa v stimata
    gts4 = []
    imgs_name = []  # conterra il nome delle immagini di tutti i batch del validation

    for batch in test_loader:
        x = Variable(batch['image'])  # immagini del batch corrente
        names = batch['name']  # nomi immagini del batch corrente
        if torch.cuda.is_available():
            x = x.cuda()
        pred = model(x).data.numpy().copy()
        gt = batch['pose'].numpy().copy()

        imgs_name.append(names)  # appendiamo i nomi delle immagini del batch corrente
        preds1.append(pred.transpose()[0])  # posa x stimata del batch
        gts1.append(gt.transpose()[0])  # posa x reale del batch
        preds2.append(pred.transpose()[1])  # posa y stimata del batch
        gts2.append(gt.transpose()[1])  # posa y reale del batch
        preds3.append(pred.transpose()[2])  # posa u stimata del batch
        gts3.append(gt.transpose()[2])  # posa u reale del batch
        preds4.append(pred.transpose()[3])  # posa v stimata del batch
        gts4.append(gt.transpose()[3])  # posa v reale del batch

    IMGS_NAME = np.concatenate(imgs_name)
    X = np.concatenate(preds1), np.concatenate(gts1)
    Y = np.concatenate(preds2), np.concatenate(gts2)
    U = np.concatenate(preds3), np.concatenate(gts3)
    V = np.concatenate(preds4), np.concatenate(gts4)

    # salviamo le predizioni e le gts in una matrice (ciascuno) N x 5 (img_name,x,y,u,v)
    preds_pose = zip(IMGS_NAME, X[0], Y[0], U[0], V[0])
    gts_pose = zip(IMGS_NAME, X[1], Y[1], U[1], V[1])

    with open(predicted_pose_str, "w") as f:
        writer = csv.writer(f)
        for elem in preds_pose:
            writer.writerow(elem)

    with open(target_pose_str, "w") as f:
        writer = csv.writer(f)
        for elem in gts_pose:
            writer.writerow(elem)

    # print X[0] # array contenente le 3101 pose x predette del test set
    # print len(IMGS_NAME)

    return (X, Y, U, V)







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







# ---- REC CURVE ----

def rec_curve(predictions, gt):
    #calcoliamo tutti gli errori
    errors = (predictions-gt)**2
    #scegliamo i valori unici e ordiniamoli per definire le soglie di tolleranza
    tolerances = np.sort(np.unique(errors))
    correct= [] #lista delle "accuracy" relative a ogni soglia
    for t in tolerances:
        correct.append((errors<=t).mean()) #frazione di elementi "correttamente" regressi
    AUC = np.trapz(correct, tolerances) #area sotto la curva calcolata col metodo dei trapezi
    tot_area = np.max(tolerances)*1 #area totale
    AOC = tot_area - AUC
    #restituiamo le soglie, la frazione di campioni correttamente regressi e l'area sopra la curva
    return tolerances, correct, AOC









def mlp():

    #----- MODEL -----
    mlp_regressor = MLPRegressor(110592, 4, 512)

    # training set
    train = Dataset('dataset/images', 'dataset/training_list.csv', transform=transforms.ToTensor())

    # ------ NORMALIZATION ---------

    m, s = normalization(train)


    # --------- DATALOADER AND TRANSFORMATION ----------

    transform = transforms.Compose([transforms.ToTensor(),  # conversione in tensore
                                    transforms.Normalize(m, s),  # nomrlizzazione con media e dvst del TS
                                    transforms.Lambda(lambda x: x.view(-1))])  # trasforma l'immagine in un unico vettore

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
    train_loader = DataLoader(train, batch_size=64, num_workers=2, shuffle=True)  # shuffle accede ai dati in maniera casuale
    valid_loader = DataLoader(valid, batch_size=64, num_workers=2)




    # ----- START TRAINING -------

    mlp_regressor, mlp_regressor_logs = train_regression(mlp_regressor, train_loader, valid_loader, epochs=100)

    # save the model
    torch.save(mlp_regressor.state_dict(), 'mlp_regressor.pth')



    # ----- PLOT LOGS--------

    plot_logs_regression(mlp_regressor_logs)

    # save plot
    plt.savefig('loss_mlp_regressor', format="jpg", bbox_inches='tight', pad_inches=0)



    # ----- MSE --------

    mlp_regressor_preds_gts = test_model_regression(mlp_regressor, valid_loader, "predicted_pose_mlp", "target_pose_mlp")

    MSE1 = metrics.mean_squared_error(mlp_regressor_preds_gts[0][1], mlp_regressor_preds_gts[0][0])  # ..reali predette..
    MSE2 = metrics.mean_squared_error(mlp_regressor_preds_gts[1][1], mlp_regressor_preds_gts[1][0])
    MSE3 = metrics.mean_squared_error(mlp_regressor_preds_gts[2][1], mlp_regressor_preds_gts[2][0])
    MSE4 = metrics.mean_squared_error(mlp_regressor_preds_gts[3][1], mlp_regressor_preds_gts[3][0])
    # print"MSE pose X:", MSE1
    # print"MSE pose Y:", MSE2
    # print"MSE pose U:", MSE3
    # print"MSE pose V:", MSE4

    # save on txt file
    with open("MSE_test_MLP_regressor.txt", "w") as text_file:
        text_file.write(
            "MSE1 MLP Regressor: {:.2f}\nMSE2 MLP Regressor: {:.2f}\nMSE3 MLP Regressor: {:.2f}\nMSE4 MLP Regressor: {:.2f}".format(MSE1, MSE2, MSE3, MSE4))




    # ---- REC CURVE ----

    # REC curve pose x
    mlp_regressor_rec1 = rec_curve(mlp_regressor_preds_gts[0][0], mlp_regressor_preds_gts[0][1])
    # REC curve pose y
    mlp_regressor_rec2 = rec_curve(mlp_regressor_preds_gts[1][0], mlp_regressor_preds_gts[1][1])
    # REC curve pose u
    mlp_regressor_rec3 = rec_curve(mlp_regressor_preds_gts[2][0], mlp_regressor_preds_gts[2][1])
    # REC curve pose v
    mlp_regressor_rec4 = rec_curve(mlp_regressor_preds_gts[3][0], mlp_regressor_preds_gts[3][1])

    plt.figure(figsize=(18, 10))
    plt.subplot(221)
    plt.title('REC Curve (x)')
    plt.plot(mlp_regressor_rec1[0], mlp_regressor_rec1[1])
    plt.legend(['MLP Regressor (x). AOC: %0.2f' % mlp_regressor_rec1[2]])
    plt.grid()

    plt.subplot(222)
    plt.title('REC Curve (y)')
    plt.plot(mlp_regressor_rec2[0], mlp_regressor_rec2[1])
    plt.legend(['MLP Regressor (y). AOC: %0.2f' % mlp_regressor_rec2[2]])
    plt.grid()

    plt.subplot(223)
    plt.title('REC Curve (u)')
    plt.plot(mlp_regressor_rec3[0], mlp_regressor_rec3[1])
    plt.legend(['MLP Regressor (u). AOC: %0.2f' % mlp_regressor_rec3[2]])
    plt.grid()

    plt.subplot(224)
    plt.title('REC Curve (v)')
    plt.plot(mlp_regressor_rec4[0], mlp_regressor_rec4[1])
    plt.legend(['MLP Regressor (v). AOC: %0.2f' % mlp_regressor_rec4[2]])
    plt.grid()

    # plt.show()

    # save plot
    plt.savefig('REC_mlp_regressor', format="jpg", bbox_inches='tight', pad_inches=0)




    # Valutiamo adesso i risultati di regressione: usare il modulo "evaluate.py"
    errors = main('predicted_pose_mlp', 'target_pose_mlp')
    # save on txt file
    with open("mlp_errors.txt", "w") as text_file:
        text_file.write("Mean Location Error: {:.4f}\nMedian Location Error: {:.4f}\nMean Orientation Error: {:.4f}\nMedian Orientation Error: {:.4f}"
                        .format(errors[0], errors[1], errors[2], errors[3]))






def deep_mlp():

    #----- MODEL -----
    deep_mlp_regressor = DeepMLPRegressor(110592, 4, 512)

    # training set
    train = Dataset('dataset/images', 'dataset/training_list.csv', transform=transforms.ToTensor())

    # ------ NORMALIZATION ---------

    m, s = normalization(train)


    # --------- DATALOADER AND TRANSFORMATION ----------

    transform = transforms.Compose([transforms.ToTensor(),  # conversione in tensore
                                    transforms.Normalize(m, s),  # nomrlizzazione con media e dvst del TS
                                    transforms.Lambda(lambda x: x.view(-1))])  # trasforma l'immagine in un unico vettore

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
    train_loader = DataLoader(train, batch_size=64, num_workers=2, shuffle=True)  # shuffle accede ai dati in maniera casuale
    valid_loader = DataLoader(valid, batch_size=64, num_workers=2)




    # ----- START TRAINING -------

    deep_mlp_regressor, deep_mlp_regressor_logs = train_regression(deep_mlp_regressor, train_loader, valid_loader, epochs=100)

    # save the model
    torch.save(deep_mlp_regressor.state_dict(), 'deep_mlp_regressor.pth')



    # ----- PLOT LOGS--------

    plot_logs_regression(deep_mlp_regressor_logs)

    # save plot
    plt.savefig('loss_deep_mlp_regressor', format="jpg", bbox_inches='tight', pad_inches=0)



    # ----- MSE --------

    deep_mlp_regressor_preds_gts = test_model_regression(deep_mlp_regressor, valid_loader, "predicted_pose_deep_mlp", "target_pose_deep_mlp")

    MSE1 = metrics.mean_squared_error(deep_mlp_regressor_preds_gts[0][1], deep_mlp_regressor_preds_gts[0][0])  # ..reali predette..
    MSE2 = metrics.mean_squared_error(deep_mlp_regressor_preds_gts[1][1], deep_mlp_regressor_preds_gts[1][0])
    MSE3 = metrics.mean_squared_error(deep_mlp_regressor_preds_gts[2][1], deep_mlp_regressor_preds_gts[2][0])
    MSE4 = metrics.mean_squared_error(deep_mlp_regressor_preds_gts[3][1], deep_mlp_regressor_preds_gts[3][0])
    # print"MSE pose X:", MSE1
    # print"MSE pose Y:", MSE2
    # print"MSE pose U:", MSE3
    # print"MSE pose V:", MSE4

    # save on txt file
    with open("MSE_test_DEEP_MLP_regressor.txt", "w") as text_file:
        text_file.write(
            "MSE1 DEEP MLP Regressor: {:.2f}\nMSE2 DEEP MLP Regressor: {:.2f}\nMSE3 DEEP MLP Regressor: {:.2f}\nMSE4 DEEP MLP Regressor: {:.2f}".format(MSE1, MSE2, MSE3, MSE4))




    # ---- REC CURVE ----

    # REC curve pose x
    deep_mlp_regressor_rec1 = rec_curve(deep_mlp_regressor_preds_gts[0][0], deep_mlp_regressor_preds_gts[0][1])
    # REC curve pose y
    deep_mlp_regressor_rec2 = rec_curve(deep_mlp_regressor_preds_gts[1][0], deep_mlp_regressor_preds_gts[1][1])
    # REC curve pose u
    deep_mlp_regressor_rec3 = rec_curve(deep_mlp_regressor_preds_gts[2][0], deep_mlp_regressor_preds_gts[2][1])
    # REC curve pose v
    deep_mlp_regressor_rec4 = rec_curve(deep_mlp_regressor_preds_gts[3][0], deep_mlp_regressor_preds_gts[3][1])

    plt.figure(figsize=(18, 10))
    plt.subplot(221)
    plt.title('REC Curve (x)')
    plt.plot(deep_mlp_regressor_rec1[0], deep_mlp_regressor_rec1[1])
    plt.legend(['DEEP MLP Regressor (x). AOC: %0.2f' % deep_mlp_regressor_rec1[2]])
    plt.grid()

    plt.subplot(222)
    plt.title('REC Curve (y)')
    plt.plot(deep_mlp_regressor_rec2[0], deep_mlp_regressor_rec2[1])
    plt.legend(['DEEP MLP Regressor (y). AOC: %0.2f' % deep_mlp_regressor_rec2[2]])
    plt.grid()

    plt.subplot(223)
    plt.title('REC Curve (u)')
    plt.plot(deep_mlp_regressor_rec3[0], deep_mlp_regressor_rec3[1])
    plt.legend(['DEEP MLP Regressor (u). AOC: %0.2f' % deep_mlp_regressor_rec3[2]])
    plt.grid()

    plt.subplot(224)
    plt.title('REC Curve (v)')
    plt.plot(deep_mlp_regressor_rec4[0], deep_mlp_regressor_rec4[1])
    plt.legend(['DEEP MLP Regressor (v). AOC: %0.2f' % deep_mlp_regressor_rec4[2]])
    plt.grid()

    # plt.show()

    # save plot
    plt.savefig('REC_deep_mlp_regressor', format="jpg", bbox_inches='tight', pad_inches=0)




    # Valutiamo adesso i risultati di regressione: usare il modulo "evaluate.py"
    main('predicted_pose_deep_mlp', 'target_pose_deep_mlp')