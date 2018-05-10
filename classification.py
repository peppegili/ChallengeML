#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from dataset import Dataset
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader

# COSTRUZIONE TRAINING SET E VALIDATION SET

# training set
train = Dataset('dataset/images','dataset/training_list.csv',transform=transforms.ToTensor())
sample = train[0]
#l'immagine è 3 x 144 x 256 perché è una immagine a colori
print "Immagine di train:", sample['image'].shape
print "Etichetta:", sample['label']
print "Posa:", sample['pose']

print ""

# validation set
valid = Dataset('dataset/images','dataset/validation_list.csv',transform=transforms.ToTensor())
sample = valid[0]
#l'immagine è 3 x 144 x 256 perché è una immagine a colori
print "Immagine di validation:", sample['image'].shape
print "Etichetta:", sample['label']
print "Posa:", sample['pose']






# EFFETTUIAMO ADESSO LA NORMALIZZAZIONE, E POI PER RIDURRE I TEMPI COMPUTAZIONALI LAVORIAMO CON IMMAGINI PIU PICCOLE, 32X56

# per il training set

#procedura per calcolare la media
m = np.zeros(3)

for sample in train:
    m+=sample['image'].sum(1).sum(1) #accumuliamo la somma dei pixel canale per canale
#dividiamo per il numero di immagini moltiplicato per il numero di pixel
m=m/(len(train)*144*256)

#procedura simile per calcolare la deviazione standard
s = np.zeros(3)

for sample in train:
    s+=((sample['image']-torch.Tensor(m).view(3,1,1))**2).sum(1).sum(1)
s=np.sqrt(s/(len(train)*144*256))

# media e devST per i 3 canali
print "Medie",m
print "Dev.Std.",s

# inseriamo la corretta normalizzazione tra le trasformazioni.
# così ogni Ogni immagine (sia di train che di validation) viene normalizzata e trasformata in un vettore.
transform = transforms.Compose([transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(m,s),
                               transforms.Lambda(lambda x: x.view(-1))]) #per trasformare l'immagine in un unico vettore
train = Dataset('dataset/images','dataset/training_list.csv',transform=transform)
print "Immagine di train:", train[0]['image'].shape # 3x32x54
print "Etichetta:", train[0]['label']
print "Posa:", train[0]['pose']

print ""

valid = Dataset('dataset/images','dataset/validation_list.csv',transform=transform)
print "Immagine di validation:", valid[0]['image'].shape # 3x32x54
print "Etichetta:", valid[0]['label']
print "Posa:", valid[0]['pose']



# PER EFFETTUARE L'OTTIMIZZAZIONE MEDIANTE SGD DOBBIAMO SUDDIVIDERE I CAMPIONI IN MINI-BATCH
# INOLTRE E IMPORTANTE FORNIRE I CAMPIONI IN ORDINE CASUALE
# Utilizziamo un batch size di 64 immagini e due thread paralleli per velocizzare il caricamento dei dati:

train_loader = DataLoader(train, batch_size=64, num_workers=2, shuffle=True)
#shuffle permette di accedere ai dati in maniera casuale
valid_loader = DataLoader(valid, batch_size=64, num_workers=2)

# I data loader sono degli oggetti iterabili. Possiamo dunque accedere ai diversi batch in maniera sequenziale all'interno di un ciclo for.
# Proviamo ad accedere al primo batch e interrompiamo il ciclo:

for batch in train_loader:
    break
print batch['image'].shape
print batch['label'].shape
print batch['pose'].shape
# il batch contiene 64 vettori di training di dimensione 5376 e altrettante etichette e pose corrispondenti



