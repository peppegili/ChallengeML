#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
import regression
import classification
from inference import*
import pandas as pd
from classification_models import *
from regression_models import*





# --------- SEED ----------
np.random.seed(1234)
torch.random.manual_seed(1234)




#-------------  MLP CLASSIFICATION -------------
#classification.mlp()


#------------- DEEP MLP CLASSIFICATION -------------
#classification.deep_mlp()


#------------ VGG16 FC fine tuning CLASSIFICATION--------------
#classification.vgg16_fc()


#------------ VGG16 CL FC fine tuning CLASSIFICATION--------------
#classification.vgg16_cl_fc()











#-------------  MLP REGRESSION -------------
#regression.mlp()


#------------- DEEP MLP REGRESSION -------------
#regression.deep_mlp()


#------------ VGG16 FC fine tuning REGRESSION--------------
#regression.vgg16_fc()


#------------ VGG16 CL FC fine tuning REGRESSION--------------
#regression.vgg16_cl_fc()










# --------------- INFERENZA --------------------

# PREDIZIONE CLASSI E POSE TEST SET
#cls_model, reg_model = load_models(MLPClassifier(110592, 16, 512), MLPRegressor(110592, 4, 512), "mlp_classifier.pth", "mlp_regressor.pth")
#predictions(cls_model, reg_model, get_test_batches(), "predictions.csv")