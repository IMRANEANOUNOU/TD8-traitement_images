from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def Test():
    print("="*50)
    print(" DÉBUT DES TESTS DU TD TRAITEMENT D'IMAGES ")
    print("="*50)

    print("\n--- Préliminaires : Chargement Image ---")
    image_lenna = None
    try:
        image_lenna = plt.imread("lenna.png")
        print("Succès : L'image 'lenna.png' a été chargée correctement.")
        print(f"Dimensions de lenna.png : {image_lenna.shape}")
    except FileNotFoundError:
        print("Attention : L'image 'lenna.png' est introuvable dans le répertoire courant.")

    matrice_test_gris = [[10, 50, 100], [200, 255, 0], [127, 127, 127]]
    matrice_test_rgb = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
    matrice_stegano = [[0, 127], [255, 0]]
    liste_stegano = [1, 0, 1, 1]

    print("\n--- Partie I : Images Noir et Blanc ---")
    print("1. Image Noire (2x2) :", image_noire(2, 2))
    print("2. Image Blanche (2x2) :", image_blanche(2, 2))
    img_bn = creerImgBlancNoir(2, 2)
    print("3. Image Noir & Blanc (Damier 2x2) :", img_bn)
    print("4. Négatif du Damier :", negatif(img_bn))

    print("\n--- Partie II : Images en Niveau de Gris ---")
    print(f"Matrice de test : {matrice_test_gris}")
    print("5. Luminance (Moyenne) :", luminance(matrice_test_gris))
    print("6. Contraste (Variance) :", constrast(matrice_test_gris))
    print("7. Profondeur (Max) :", profondeur(matrice_test_gris))

    print("\n--- Partie III : Opérations Élémentaires ---")
    print("9. Inverser :", inverser(matrice_test_gris))
    print("10. Symétrie horizontale (flipH) :", flipH(matrice_test_gris))
    print("11. Poser Verticalement :", poserV([[1,1]], [[2,2]]))
    print("12. Poser Horizontalement :", poserH([[1,1]], [[2,2]]))

    print("\n--- Partie IV : Transferts ---")
    hist = histogramme(matrice_test_gris)
    print("15. Histogramme (extrait, premières valeurs) :", hist[:10], "...")
    print("16. Égaliser :", egaliser(matrice_test_gris))
    print("18. Réduire (vers profondeur 16) :", reduire(matrice_test_gris, 16))

    print("\n--- Partie V : Images RGB ---")
    try:
        print("21. Initialisation Aléatoire (2x2) :", initImageRGB(2, 2))
        print("22. Symétrie RGB :", symetrie(matrice_test_rgb))
        print("23. Grayscale (Conversion Niveaux de gris) :", grayscale(matrice_test_rgb))
    except NameError:
        print("Fonctions RGB non définies dans le script actuel.")

    print("\n--- Partie VI : Stéganographie ---")
    try:
        matrice_C = disposer(liste_stegano, matrice_stegano)
        print("15. Disposer (Matrice C) :", matrice_C)
        matrice_B = dissimuler(matrice_stegano, matrice_C)
        print("17. Dissimuler (Matrice B) :", matrice_B)
        print("18. Retrouver le message :", retrouver(matrice_B))
    except NameError:
        print("Fonctions de Stéganographie non définies dans le script actuel.")

    print("\n--- Partie VII : Segmentation d'image ---")
    try:
        print("19. Segmentation manuelle (Seuil=100) :", segmentation(matrice_test_gris, 100))
        seuil_auto = seuil_automatique(matrice_test_gris, 1.0)
        print(f"20. Seuil automatique trouvé (eps=1.0) : {seuil_auto}")
        print("20. Segmentation automatique :", segmentation_automatique(matrice_test_gris))
    except NameError:
        print("Fonctions de Segmentation non définies dans le script actuel.")

    print("\n" + "="*50)
    print(" FIN DES TESTS ")
    print("="*50)


def image_noire(h, l):
    return [[0 for j in range(l)] for i in range(h)]

def image_blanche(h, l):
    return [[1 for j in range(l)] for i in range(h)]

def creerImgBlancNoir(h, l):
    return [[(i + j) % 2 for j in range(l)] for i in range(h)]

def negatif(Img):
    return [[1 if pixel == 0 else 0 for pixel in ligne] for ligne in Img]


def luminance(Img):
    if not Img or not Img[0]:
        return 0
    lignes = len(Img)
    colonnes = len(Img[0])
    somme = sum(sum(ligne) for ligne in Img)
    N = lignes * colonnes
    return somme / N

def constrast(Img):
    if not Img or not Img[0]:
        return 0
    lignes = len(Img)
    colonnes = len(Img[0])
    N = lignes * colonnes
    moy = luminance(Img)
    
    somme_variance = 0
    for i in range(lignes):
        for j in range(colonnes):
            somme_variance += (Img[i][j] - moy) ** 2
            
    return somme_variance / N


def profondeur(Img):
    if not Img or not Img[0]:
        return 0
    max_val = 0
    for ligne in Img:
        for pixel in ligne:
            if pixel > max_val:
                max_val = pixel
    return max_val


def Ouvrir(chemin_image):
    img = Image.open(chemin_image)
    img_gray = img.convert('L')
    matrice = np.array(img_gray)
    return matrice.tolist()


def inverser(img):
    if not img or not img[0]:
        return []
    p = profondeur(img)
    return [[p - pixel for pixel in ligne] for ligne in img]

def flipH(img):
    return [ligne[::-1] for ligne in img]

def poserV(img1, img2):
    return img1 + img2

def poserH(img1, img2):
    if not img1 or not img2 or len(img1) != len(img2):
        return []
    
    img_resultante = []
    for i in range(len(img1)):
        nouvelle_ligne = img1[i] + img2[i]
        img_resultante.append(nouvelle_ligne)
        
    return img_resultante


def transferer(img, q, t):
    if not img or not img[0]:
        return []
    
    img_resultante = []
    for ligne in img:
        nouvelle_ligne = []
        for pixel in ligne:
            nouveau_pixel = t[pixel]
            nouvelle_ligne.append(nouveau_pixel)
        img_resultante.append(nouvelle_ligne)
        
    return img_resultante


def inverser_transfert(img):
    if not img or not img[0]:
        return []
    
    p = profondeur(img) 
    q = p
    
    t = [p - v for v in range(p + 1)]
    
    return transferer(img, q, t)


def histogramme(img):
    if not img or not img[0]:
        return []
        
    p = profondeur(img)
    hist = [0] * (p + 1)
    
    for ligne in img:
        for pixel in ligne:
            hist[pixel] += 1
            
    return hist


def egaliser(img):
    if not img or not img[0]:
        return []
    
    h = len(img)
    l = len(img[0])
    N = h * l
    
    p = profondeur(img)
    hi = histogramme(img)
    
    vmin = 0
    while vmin <= p and hi[vmin] == 0:
        vmin += 1
        
    if N == hi[vmin]:
        return img 
        
    denominateur = N - hi[vmin]
    
    cumul_hi = [0] * (p + 1)
    somme_en_cours = 0
    for k in range(p + 1):
        somme_en_cours += hi[k]
        cumul_hi[k] = somme_en_cours
        
    t = [0] * (p + 1)
    for v in range(vmin, p + 1):
        v_prime = p * (cumul_hi[v] - hi[vmin]) / denominateur
        t[v] = int(round(v_prime, 0))
        
    return transferer(img, p, t)


def reduire(img, q):
    if not img or not img[0]:
        return []
        
    p = profondeur(img)
    
    t = [0] * (p + 1)
    for v in range(p + 1):
        v_prime = int(round((v * q) / p, 0))
        t[v] = v_prime
        
    return transferer(img, q, t)


def segmentation(A, v):
    if not A or not A[0]:
        return []
    return [[0 if pixel <= v else 255 for pixel in ligne] for ligne in A]


def histogramme(A):
    H = [0] * 256
    for ligne in A:
        for pixel in ligne:
            if 0 <= pixel <= 255:
                H[pixel] += 1
    return H

def seuil_automatique(A, epsilon):
    valeurs = [pixel for ligne in A for pixel in ligne]
    
    if not valeurs:
        return 0
        
    min_A = min(valeurs)
    max_A = max(valeurs)
    
    v = (min_A + max_A) / 2.0
    
    while True:
        inf_v = [p for p in valeurs if p < v]
        sup_v = [p for p in valeurs if p >= v]
        
        u1 = sum(inf_v) / len(inf_v) if len(inf_v) > 0 else 0
        u2 = sum(sup_v) / len(sup_v) if len(sup_v) > 0 else 0
        
        v_prime = (u1 + u2) / 2.0
        
        if abs(v - v_prime) <= epsilon:
            return v_prime
        
        v = v_prime

def segmentation_automatique(A):
    epsilon = 1.0 
    seuil_optimal = seuil_automatique(A, epsilon)
    return segmentation(A, seuil_optimal)

Test()
