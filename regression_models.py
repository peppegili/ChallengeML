#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from torchvision import models




# --------- MLP ---------  1 hidden layer

class MLPRegressor(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        """Costruisce un classificatore MLP.
            Input:
                in_features: numero di feature in input (es. 110592)
                out_classes: numero di features in uscita (es. 4)
                hidden_units: numero di unità nel livello nascosto (es. 512)"""
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_features, hidden_units),
                                   nn.Tanh(),
                                   nn.Linear(hidden_units, out_features))

    def forward(self, x):
        """Definisce come processare l'input x"""
        return self.model(x)






# --------- DEEP MLP ---------  2 hidden layer

class DeepMLPRegressor(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        """Costruisce un regressore MLP.
            Input:
                in_features: numero di feature in input (es. 110592)
                out_classes: numero di feature in uscita (es. 4)
                hidden_units: numero di unità nel livello nascosto (es. 512)"""
        super(DeepMLPRegressor, self).__init__()

        self.model = nn.Sequential(nn.Linear(in_features, hidden_units),  # livello nascosto 1
                                   nn.Tanh(),
                                   nn.Linear(hidden_units, hidden_units),  # livello nascosto 2
                                   nn.Tanh(),
                                   nn.Linear(hidden_units, out_features))  # livello di uscita

    def forward(self, x):
        return self.model(x)






#---- VGG16 ----

def vgg16_model():
    model = models.vgg16(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False  # freeze dei layer convoluzionali

    layer = -1
    for child in model.features.children():
        # print child
        layer += 1
        if layer > 16 and layer < 31:
            # print"Sfreezato -> ", child
            for param in child.parameters():
                param.requires_grad = True

    # cambiamo i dropout nel blocco fully connected
    for layer in model.classifier.children():
        if (type(layer) == nn.Dropout):
            layer.p = 0.25

    features = list(model.classifier.children())[1:-1]  # rimozione primo e ultimo layer

    del features[2]  # rimozione quarto livello

    features.insert(0, nn.Linear(8192, 2048))  # aggiungiamo il primo layer # img 144x256
    features.insert(3, nn.Linear(2048, 2048))  # aggiungiamo il quarto layer
    features.append(nn.Linear(2048, 4))  # aggiungiamo layer con 16 output

    model.classifier = nn.Sequential(*features)  # sostituiamo il modulo "classifier"

    # print model

    return model