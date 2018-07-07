#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
from PIL import Image
from os import path
import numpy as np


# CREAZIONE DELLA CLASSE CHE CI PERMETTE DI CARICARE LE IMMAGINI, X, Y, U, V E LE RELATIVE ETICHETTE

class Dataset(Dataset):
    """Implementa l'oggetto ClassificationDataset che ci permette di caricare
    le immagini del dataset images"""

    def __init__(self, base_path, csv_list, transform=None):
        """Input:
            base_path: il path alla cartella contenente le immagini
            txt_list: il path al file di testo contenente la lista delle immagini
                        con le relative etichette. Ad esempio train.csv o test.csv.
            transform: implementeremo il dataset in modo che esso supporti le trasformazioni"""
        # conserviamo il path alla cartella contenente le immagini
        self.base_path = base_path
        # carichiamo la lista dei file
        # sarà una matrice con n righe (numero di immagini) e 2 colonne (path, etichetta)
        # self.images = np.loadtxt(csv_list, dtype=str, delimiter=',', usecols=(0,5))
        # sarà una matrice con n righe (numero di immagini) e 6 colonne (path, x, y, u, v, etichetta)
        self.images = np.loadtxt(csv_list, dtype=str, delimiter=',')

        # conserviamo il riferimento alla trasformazione da applicare
        self.transform = transform


    def __getitem__(self, index):
        # recuperiamo il path dell'immagine di indice index e la relativa etichetta
        # f, c = self.images[index]
        # recuperiamo il path dell'immagine di indice index, x, y, u, v e la relativa etichetta
        f, x, y, u, v, c = self.images[index]

        # carichiamo l'immagine utilizzando PIL
        im = Image.open(path.join(self.base_path, f))

        # se la trasfromazione è definita, applichiamola all'immagine
        if self.transform is not None:
            im = self.transform(im)

        # convertiamo l'etichetta in un intero
        label = int(c)

        # restituiamo un dizionario contenente immagine etichetta
        # return {'image': im, 'label': label}
        # restituiamo un dizionario contenente nome immagine - immagine - etichetta - posa
        return {'name': f, 'image': im, 'label': label, 'pose': np.array([x, y, u, v], dtype='float')}

    # restituisce il numero di campioni: la lunghezza della lista "images"
    def __len__(self):
        return len(self.images)
