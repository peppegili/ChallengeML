#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn





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