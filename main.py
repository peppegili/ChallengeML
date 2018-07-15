#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
import regression
import classification
from inference import*
import pandas as pd
import classification_models
import regression_models
from torchvision import models




# --------- SEED ----------
np.random.seed(1234)
torch.random.manual_seed(1234)





# ---------- METODO PROPOSTO PER LA CLASSIFICAZIONE --------

#-----VGG16 CL FC fine tuning CLASSIFICATION - DATA AUGMENTATION-----
#classification.vgg16_cl_fc_aug()




# ----------- METODO PROPOSTO PER LA REGRESSIONE ----------

#-----VGG16 CL FC fine tuning REGRESSION - DATA AUGMENTATION-----
#regression.vgg16_cl_fc_aug()






# --------------- INFERENZA --------------------

# PREDIZIONE CLASSI E POSE TEST SET
#vgg16_cls_model = classification_models.vgg16_model()
#vgg16_reg_model = regression_models.vgg16_model()

#cls_model, reg_model = load_models(vgg16_cls_model, vgg16_reg_model, "vgg16_classifier.pth", "vgg16_regressor.pth")
#predictions(cls_model, reg_model, get_test_batches(), "predictions.csv")





















#---------------------------- ALTRi METODI TESTATI ------------------------------------


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