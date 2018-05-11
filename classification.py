#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from dataset import Dataset
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


# --------------------------------- FUNZIONI ---------------------------------------

# classificatore SoftMax
class SoftMaxRegressor(nn.Module):
    def __init__(self, in_features, out_classes):
        """Costruisce un regressore softmax.
            Input:
                in_features: numero di feature in input (es. 4)
                out_classes: numero di classi in uscita (es. 3)"""
        super(SoftMaxRegressor, self).__init__() #richiamo il costruttore della superclasse
        #questo passo è necessario per abilitare alcuni meccanismi automatici dei moduli di PyTorch
        self.linear = nn.Linear(in_features,out_classes) #il regressore softmax restituisce
        #distribuzioni di probabilità, quindi il numero di feature di output coincide con il numero di classi
        self.softmax = nn.Softmax(dim=1) #dim=1 indica che il softmax verrà calcolato riga per riga
    def forward(self,x):
        """Definisce come processare l'input x"""
        scores = self.linear(x)
        #anche in questo caso vogliamo evitare di applicare il softmax in fase di training
        if self.training: #se il modulo è in fase di training
            #la proprietà "training" è messa a disposizione dai meccanismi
            #automatici dei moduli di PyTorch
            return scores
        else: #se siamo in fase di test, calcoliamo le probabilità
            return self.softmax(scores)

# ----------------------------------------------------------------------------------








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
print "Immagine di validation:", valid[0]['image'].shape # 3x32x56
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

for batch in train_loader: #161 batch dove ogni batch contiene 64 immagini
    break
print batch['image'].shape
print batch['label'].shape
print batch['pose'].shape
# il batch contiene 64 vettori di training (immagini) di dimensione 5376 e altrettante etichette e pose corrispondenti




# ALLENAMENTO MEDIANTE SGD

lr = 0.001
epochs = 10
#5376 feature in ingresso e 16 classi
model = SoftMaxRegressor(5376,16)
criterion = nn.CrossEntropyLoss()
#l'optimizer ci permetterà di effettuare la Stochastic Gradient Descent
optimizer = SGD(model.parameters(),lr)
training_losses = []
training_accuracies = []
valid_losses = []
valid_accuracies = []
for e in range(epochs):
    #ciclo di training
    model.train()
    train_loss = 0
    train_acc = 0
    for i, batch in enumerate(train_loader):
        #trasformiamo i tensori in variabili
        x=Variable(batch['image'])
        y=Variable(batch['label'])
        output = model(x)
        l = criterion(output,y)
        l.backward()
        acc = accuracy_score(y.data,output.max(1)[1].data)
        #accumuliamo i valori di training e loss
        #moltiplichiamo per x.shape[0], che restituisce la dimensione
        #del batch corrente.
        train_loss+=l.data[0]*x.shape[0]
        train_acc+=acc*x.shape[0]
        print "\r[TRAIN] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" %  \
                (e+1,epochs,i,len(train_loader),l.data[0],acc),
        optimizer.step() #sostituisce il codice di aggiornamento manuale dei parametri
        optimizer.zero_grad() #sostituisce il codice che si occupava di azzerare i gradienti
    train_loss/=len(train)
    train_acc/=len(train)
    training_losses.append(train_loss)
    training_accuracies.append(train_acc)
    print "\r[TRAIN] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" %  \
                (e+1,epochs,i,len(train_loader),train_loss,train_acc)
    #ciclo di test
    model.eval()
    valid_acc=0
    valid_loss=0
    for i, batch in enumerate(valid_loader):
        #trasformiamo i tensori in variabili
        x=Variable(batch['image'], requires_grad=False)
        y=Variable(batch['label'], requires_grad=False)
        output = model(x)
        l = criterion(output,y)
        valid_acc += accuracy_score(y.data,output.max(1)[1].data)*x.shape[0]
        valid_loss += l.data[0]*x.shape[0]
        print "\r[VALID] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" %  \
                (e+1,epochs,i,len(valid_loader),l.data[0],acc),
    #salviamo il modello
    torch.save(model.state_dict(),'model-%d.pth'%(e+1,))
    valid_loss/=len(valid)
    valid_acc/=len(valid)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)
    print "\r[VALID] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" %  \
                (e+1,epochs,i,len(valid_loader),valid_loss,valid_acc)



# plottiamo accuracy e loss di training e validation
plt.figure(figsize=(18,6))
plt.subplot(121)
plt.plot(training_losses)
plt.plot(valid_losses)
plt.legend(['Training Loss','Validation Losses'])
plt.grid()
plt.subplot(122)
plt.plot(training_accuracies)
plt.plot(valid_accuracies)
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.grid()
plt.show()

