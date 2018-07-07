#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataset_inference
import dataset
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from classification import normalization
from classification_models import*
from regression_models import *
import csv




def get_test_batches(isVGG16=False):

    # se non utilizziamo il modello vgg16 dobbiamo normalizzare rispetto alle statistiche del TS
    if isVGG16 == False:
        print "Inferenza: il modello utilizzato non è VGG16"
        # get mean and devST of training data
        train = dataset.Dataset('dataset/images', 'dataset/training_list.csv', transform=transforms.ToTensor())

        m, s = normalization(train)

        transform = transforms.Compose([transforms.ToTensor(),  # conversione in tensore
                                        transforms.Normalize(m, s),  # nomrlizzazione con media e dvst del TS
                                        transforms.Lambda(lambda x: x.view(-1))])  # trasforma l'immagine in un unico vettore



        test = dataset_inference.Dataset('dataset/images', 'dataset/testing_list_blind.csv', transform=transform)

        #print "Nome immagine:", test[1]['name']
        #print "Immagine di test:", test[1]['image'].shape # 3x144x256


    # altrimenti dobbiamo normalizzare rispetto alle statistica di vgg16
    else:
        print "Inferenza: il modello utilizzato è VGG16"
        transform = transforms.Compose([transforms.ToTensor(),  # conversione in tensore
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # nomrlizzazione con media e dvst del TS


        test = dataset_inference.Dataset('dataset/images', 'dataset/testing_list_blind.csv', transform=transform)



    # dataloader
    test_loader = DataLoader(test, batch_size=64, num_workers=2)

    return test_loader



def load_model(model_obj, model_name):
    model = model_obj
    model.load_state_dict(torch.load(model_name))
    return model



def predict_classes(model, test_loader, predictions_classes_str):
    softmax = nn.Softmax(dim=1)
    model.eval()
    preds = []
    imgs_name = []  # conterra il nome delle immagini di tutti i batch del test

    for batch in test_loader:
        x=Variable(batch['image'])
        names = batch['name']  # nomi immagini del batch corrente

        #applichiamo la funzione softmax per avere delle probabilità
        if torch.cuda.is_available():
            x = x.cuda()
        pred = softmax(model(x)).data.cpu().numpy().copy()


        preds.append(pred)
        imgs_name.append(names)  # appendiamo i nomi delle immagini del batch corrente



    predictions = zip(np.concatenate(imgs_name), np.concatenate(preds).argmax(1))

    with open(predictions_classes_str, "w") as f:
        writer = csv.writer(f)
        for elem in predictions:
            writer.writerow(elem)



def predict_poses(model, test_loader, predictions_poses_str):
    model.eval()
    preds1 = []  # conterra la posa x stimata
    preds2 = []  # conterra la posa y stimata
    preds3 = []  # conterra la posa u stimata
    preds4 = []  # conterra la posa v stimata
    imgs_name = []  # conterra il nome delle immagini di tutti i batch del test

    for batch in test_loader:
        x=Variable(batch['image'])
        names = batch['name']  # nomi immagini del batch corrente

        #applichiamo la funzione softmax per avere delle probabilità
        if torch.cuda.is_available():
            x = x.cuda()
        pred = model(x).data.cpu().numpy().copy()



        imgs_name.append(names)  # appendiamo i nomi delle immagini del batch corrente
        preds1.append(pred.transpose()[0])  # posa x stimata del batch
        preds2.append(pred.transpose()[1])  # posa y stimata del batch
        preds3.append(pred.transpose()[2])  # posa u stimata del batch
        preds4.append(pred.transpose()[3])  # posa v stimata del batch




    predictions = zip(np.concatenate(imgs_name), np.concatenate(preds1), np.concatenate(preds2),
                      np.concatenate(preds3), np.concatenate(preds4))

    with open(predictions_poses_str, "w") as f:
        writer = csv.writer(f)
        for elem in predictions:
            writer.writerow(elem)









# PREDIZIONE CLASSI TEST SET
predict_classes(load_model(MLPClassifier(110592, 16, 512), "mlp_classifier.pth"), get_test_batches(isVGG16=False), "predictions_classes.csv")

# PREDIZIONE POSE TEST SET
predict_poses(load_model(MLPRegressor(110592, 4, 512), "mlp_regressor.pth"), get_test_batches(isVGG16=False), "predictions_poses.csv")