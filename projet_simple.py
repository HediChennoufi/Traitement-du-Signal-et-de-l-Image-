import numpy as np
import math
import matplotlib.pyplot as plt
import time

# Paramètres de compression
SEUIL = 6  # Valeur par défaut
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 13, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])
#Q = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], [24, 26, 56, 99, 99, 99, 99, 99],[47, 66, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99]])
#Q = Q*5
errors = []
taux = []

start = time.process_time()

# Lecture et normalisation
image = plt.imread("image.png")
taille = np.shape(image)
x = taille[0]
y = taille[1]
multiple_8_x = (x // 8) * 8
multiple_8_y = (y // 8) * 8
image = image[:multiple_8_x, :multiple_8_y]
image_finale = np.zeros((multiple_8_x, multiple_8_y, 3))

# Définition de P
P = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        P[i, j] = math.cos(((2 * j + 1) * i * math.pi) / 16) / 2
        if i == 0:
            P[i, j] = P[i, j] / math.sqrt(2)

# Compression
if(image.ndim == 3):
    for n in range(3):
        canal = image[:, :, n].astype(float)
        canal = (canal / np.max(canal)) * 255
        canal = canal - 128 * np.ones((multiple_8_x, multiple_8_y))  # [-128, 127]

        # Changement de base D = P * M * Pt
        for k in range(0, multiple_8_x, 8):
            for w in range(0, multiple_8_y, 8):
                canal[k:k+8, w:w+8] = np.matmul(P, np.matmul(canal[k:k+8, w:w+8], np.transpose(P)))
                canal[k:k+8, w:w+8] = np.divide(canal[k:k+8, w:w+8], Q).astype(int)

        # On retire le bruit en bas à droite de chaque bloc 8x8
        for k in range(0, multiple_8_x, 8):
            for w in range(0, multiple_8_y, 8):
                for z in range(0, 8):
                    for t in range(0, 8):
                        if z + t >= 16-SEUIL:
                            canal[k + z, w + t] = 0

        #Taux de compression par couche
        taux1 = (100 - (np.count_nonzero(canal)/(x * y)) * 100)
        taux.append(taux1)

        # Décompression/Récomposition
        for k in range(0, multiple_8_x, 8):
            for w in range(0, multiple_8_y, 8):
                canal[k:k+8, w:w+8] = canal[k:k+8, w:w+8] * Q
                canal[k:k+8, w:w+8] = np.matmul(np.transpose(P), np.matmul(canal[k:k+8, w:w+8], P))

        # Post processing
        canal = canal + 128
        canal = np.clip(canal, 0, 255) / 255
        image_finale[:, :, n] = canal

        # Erreur par couche
        error1 = np.linalg.norm(image[:, :, n] - canal)
        errors.append(error1 / np.linalg.norm(image[:, :, n]))

    end = time.process_time()
    em = 0
    t = 0
    for to in taux:
        t += to 
    for error in errors:
        em += error
    print("Taux de compression : ",t/3,"%" )
    print("Erreur moyenne : ", em / 3)
    print(f"Temps CPU utilisé : {end - start} secondes")
    plt.imsave('image_decompressee.png', image_finale)
    plt.imshow(image_finale)
    plt.show()
        

