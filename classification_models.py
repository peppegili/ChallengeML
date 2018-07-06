#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn





# --------- MLP ---------  1 hidden layer

class MLPClassifier(nn.Module):
    def __init__(self, in_features, out_classes, hidden_units):
        """Costruisce un classificatore MLP.
            Input:
                in_features: numero di feature in input (es. 110592)
                out_classes: numero di classi in uscita (es. 16)
                hidden_units: numero di unità nel livello nascosto (es. 512)"""
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, out_classes))

    def forward(self, x):
        """Definisce come processare l'input x"""
        scores = self.model(x)
        return scores








# ---- MLP DEEP ----  2 hidden layer

class DeepMLPClassifier(nn.Module):
    def __init__(self, in_features, out_classes, hidden_units):
        """Costruisce un classificatore MLP "profondo".
            Input:
                in_features: numero di feature in input (es. 110592)
                out_classes: numero di classi in uscita (es. 16)
                hidden_units: numero di unità nei livelli nascosti (es. 512)"""
        super(DeepMLPClassifier, self).__init__()

        self.model = nn.Sequential(nn.Linear(in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, hidden_units),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, out_classes))

    def forward(self, x):
        """Definisce come processare l'input x"""
        return self.model(x)






#---- VGG16 ----