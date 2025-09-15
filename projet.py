import numpy as np
import math
import matplotlib.pyplot as plt
import time
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Paramètres de compression
SEUIL = 6  # Valeur par défaut
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 13, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])
errors = []
taux = []

# Fonction pour charger et traiter l'image
def traiter_image(image_path):
    start = time.process_time()

    # Lecture et normalisation
    image = plt.imread(image_path)
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
            taux1 = (100 - (np.count_nonzero(canal)/(multiple_8_x * multiple_8_y)) * 100)
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
        
        return image, image_finale, t/3, em/3, end-start

# Fonction pour redimensionner l'image en respectant ses proportions
def redimensionner_image(image, max_size=600):
    largeur, hauteur = image.size
    ratio = min(max_size / largeur, max_size / hauteur)
    new_width = int(largeur * ratio)
    new_height = int(hauteur * ratio)
    return image.resize((new_width, new_height))

# Fonction pour afficher l'image avant et après
def afficher_images():
    global SEUIL
    try:
        SEUIL = int(entry_seuil.get())  # Récupérer le seuil depuis l'entrée
    except ValueError:
        print("Entrée invalide pour le seuil, utilisation de la valeur par défaut :", SEUIL)

    image_path = filedialog.askopenfilename(title="Choisir une image", filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if image_path:
        # Traitement de l'image
        image, image_finale, taux_compression, erreur_finale, temps = traiter_image(image_path)

        # Gestion des images avant et après
        img_before = Image.open(image_path)
        img_before = redimensionner_image(img_before, 400)
        img_before_tk = ImageTk.PhotoImage(img_before)

        img_after = Image.fromarray((image_finale * 255).astype(np.uint8))
        img_after = redimensionner_image(img_after, 400)
        img_after_tk = ImageTk.PhotoImage(img_after)

        label_before.config(image=img_before_tk)
        label_before.image = img_before_tk
        label_after.config(image=img_after_tk)
        label_after.image = img_after_tk

        # Mise à jour des informations de compression
        label_taux_compression.config(text=f"Taux de compression : {taux_compression:.2f} %")
        label_erreur_finale.config(text=f"Erreur finale : {erreur_finale:.4f}")
        label_temps_execution.config(text=f"Temps d'exécution : {temps:.2f} secondes")


# Création de la fenêtre Tkinter
root = Tk()
root.attributes("-fullscreen", True)
root.title("Compression d'image")

background_image = Image.open("polytech.png")
background_image = background_image.resize((600, 400))
background_tk = ImageTk.PhotoImage(background_image)
background_label = Label(root, image=background_tk)
background_label.place(relwidth=1, relheight=1)

frame = Frame(root)
frame.pack(pady=20)
info_frame = Frame(root)
info_frame.pack(pady=20)

label_taux_compression = Label(info_frame, text="Taux de compression : N/A", anchor="w")
label_taux_compression.pack(padx=5, pady=5)

label_erreur_finale = Label(info_frame, text="Erreur finale : N/A", anchor="w")
label_erreur_finale.pack(padx=5, pady=5)

label_temps_execution = Label(info_frame, text="Temps d'exécution : N/A", anchor="w")
label_temps_execution.pack(padx=5, pady=5)

Label(frame, text="Seuil de compression :").pack(side=LEFT, padx=5)
entry_seuil = Entry(frame, width=5)
entry_seuil.insert(0, str(SEUIL))
entry_seuil.pack(side=LEFT, padx=5)

button = Button(root, text="Choisir une image", command=afficher_images)
button.pack(padx=20, pady=20, expand=True)

label_before = Label(root)
label_before.pack(side=LEFT, padx=10)

label_after = Label(root)
label_after.pack(side=RIGHT, padx=10)

def quitter(event):
    root.quit()
root.bind("<Escape>", quitter)
root.mainloop()
