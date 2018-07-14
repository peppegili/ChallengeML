#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataset_inference
#import dataset
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
#from classification import normalization
from torch import nn
import csv
import time




def get_test_batches():
    transform = transforms.Compose([transforms.CenterCrop(128),  # square crop 128x128
                                    transforms.ToTensor(),  # conversione in tensore
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test = dataset_inference.Dataset('dataset/images', 'dataset/testing_list_blind.csv', transform=transform)
    #print "Nome immagine:", test[1]['name']
    #print "Immagine di test:", test[1]['image'].shape # 3x144x256

    # dataloader
    test_loader = DataLoader(test, batch_size=64, num_workers=2)

    return test_loader



def load_models(model_obj_cls, model_obj_reg, model_cls_name, model_reg_name):
    model_cls = model_obj_cls
    model_reg = model_obj_reg

    model_cls.load_state_dict(torch.load(model_cls_name))
    model_reg.load_state_dict(torch.load(model_reg_name))


    return model_cls, model_reg



def predictions(model_cls, model_reg, test_loader, predictions_str):
    since = time.time()
    print "---- START INFERENCE ----"
    softmax = nn.Softmax(dim=1)
    model_cls.eval()
    model_reg.eval()

    preds = [] # classi
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
        pred_cls = softmax(model_cls(x)).data.cpu().numpy().copy() #classi predette
        pred_reg = model_reg(x).data.cpu().numpy().copy() # pose predette



        imgs_name.append(names)  # appendiamo i nomi delle immagini del batch corrente
        preds.append(pred_cls) #classi
        preds1.append(pred_reg.transpose()[0])  # posa x stimata del batch
        preds2.append(pred_reg.transpose()[1])  # posa y stimata del batch
        preds3.append(pred_reg.transpose()[2])  # posa u stimata del batch
        preds4.append(pred_reg.transpose()[3])  # posa v stimata del batch



    predictions = zip(np.concatenate(imgs_name), np.concatenate(preds1), np.concatenate(preds2),
                      np.concatenate(preds3), np.concatenate(preds4), np.concatenate(preds).argmax(1))


    with open(predictions_str, "wb") as f:
        writer = csv.writer(f)
        for elem in predictions:
            writer.writerow(elem)
    f.close()


    time_elapsed = time.time() - since

    print('---- INFERENCE COMPLETE in {:.0f}m {:.0f}s ----'.format(time_elapsed // 60, time_elapsed % 60))

