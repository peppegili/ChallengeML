#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
import regression
import classification





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
