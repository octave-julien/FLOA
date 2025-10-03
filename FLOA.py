import numpy as np
import cv2

BANDWIDTH_FOR_MERGING_BREAKS = 15
PAGE_EDGES_COLOR = (0,255,0)

def give_projection_profile(image, orientation): #donneCourbeDensite(image, axe):
    if orientation == "H":
        projection_profile = np.average(image, axis=0)
    if orientation == "V":
        projection_profile = np.average(image, axis=1)
    projection_profile = np.round(projection_profile)
    return projection_profile
# TODO : vérifier que orientation est bien donnée

def bw_thresholding(x, threshold):
    if x <= threshold:
        return (0)
    else:
        return (255)

def binarize_projection_profile(projection_profile, threshold): #seuilleCourbeDensite(courbeDensite, seuil):
    # lumMoyenne = np.average(courbeDensite)
    # seuil : lumMoyenne = np.median(courbeDensite) # changer le nom si je garde la médiane comme seuil
    binarized_projection_profile = [bw_thresholding(x, threshold) for x in projection_profile]
    return binarized_projection_profile




def smooth_projection_profile(projection_profile, bandwidth = 21): #lisseCourbe(courbe, fenetre = 21):
    length = len(projection_profile)    #longueurCourbe = len(courbe)
    half_bandwidth = bandwidth//2       #demiFenetre = fenetre//2
    start_of_smoothing = -(-bandwidth//2)-1 # debutLissage = -(-fenetre//2)-1
    end_of_smoothing = length - start_of_smoothing #finLissage = longueurCourbe - debutLissage
    smoothed_projection_profile = projection_profile.copy()  #courbeLissee = courbe.copy()
    for i in range(start_of_smoothing, end_of_smoothing):
        smoothed_projection_profile[i] = sum(projection_profile[i-half_bandwidth:i+half_bandwidth+1])//half_bandwidth # on ne garde que la partie entière pour les performances
    return smoothed_projection_profile

def find_breaks(binarized_projection_profile, bandwidth=100, min_width_of_blank_strip = 15, min_width_of_black_strip = 50, max_width_of_black_line = 15): #largeurMinBandeBlanche = 15, largeurMinBandeNoire = 50, largeurMaxLigneNoire = 15):
    length = len(binarized_projection_profile)

    i = 0
    strip_start = 0
    strip_color = binarized_projection_profile[i]
    strips = []
    while i < length - 1:
        i = i + 1
        if binarized_projection_profile[i] == strip_color:
            continue
        else:
            strip_end = i - 1
            strip_color = binarized_projection_profile[strip_end]
            strips.append([strip_start, strip_end, strip_color, ""])
            strip_start = i
            strip_color = binarized_projection_profile[i]
    # On clot l'intervalle quand on arrive à la fin de la courbe
    strip_end = i
    strip_color = binarized_projection_profile[strip_end]
    strips.append([strip_start, strip_end, strip_color, ""])

    number_of_strips = len(strips)

    # Recherche des bandes noirs et blancs
    for i in range(number_of_strips):
        strip_width = strips[i][1] - strips[i][0] + 1
        strip_color = strips[i][2]
        if strip_width > min_width_of_black_strip and strip_color == 0:
            strips[i][3] = "bande noire"
        elif strip_width > min_width_of_blank_strip and strip_color == 255:
            strips[i][3] = "bande blanche"

    # Recherche des lignes noires
    for i in range(number_of_strips):
        if i == 0 or i == number_of_strips - 1:  # Si on se situe au premier ou au dernier intervalle:
            continue
        strip_width = strips[i][1] - strips[i][0] + 1
        strip_color = strips[i][2]
        if strip_width < max_width_of_black_line and strip_color == 0 and strips[i - 1][
            3] == "bande blanche" and strips[i + 1][3] == "bande blanche":
            strips[i][3] = "ligne noire"

    return strips

def find_edges_of_page(v_breaks, h_breaks):
    left_edge = v_breaks[1][0]
    right_edge = max([v_break[0] for v_break in v_breaks])
    top_edge = h_breaks[1][0]
    bottom_edge = max([h_break[0] for h_break in h_breaks])
    return (top_edge, right_edge, bottom_edge, left_edge)

def find_text_blocks(h_breaks, v_breaks): # Vérifier que la dénomination des limites est cohérente avec la fonction précédente. Eventuellement réordonner les arguments
    left_edges = []
    right_edges = []
    top_edges = []
    bottom_edges = []
    max_top_edge = 100000  # plus petite (en coordonnée) limite haute = marge supérieure
    min_top_edge = 0  # plus grande en coordonnée limite basse = marge inférieure
    for h_break in h_breaks:
        if h_break[3] == "bande noire":
            left_edges.append(h_break[0])
            right_edges.append(h_break[1])
    for v_break in v_breaks:
        if v_limit[3] == "bande noire":
            top_edges.append(v_break[0])
            bottom_edges.append(v_break[1])
            if v_break[0] < limiteHauteSup:
                max_top_edge = v_break[0]
            if v_break[1] > limiteBasseInf:
                min_top_edge = v_break[1]
    #nbBlocsH = len(limitesTextesGauche)
    #nbBlocsV = len(limitesTextesHaut)
    list_of_blocks = []

    for i, left_edge in enumerate(left_edges):
        new_text_block = (left_edge, max_top_edge, right_edges[i], min_top_edge)
        list_of_blocks.append(new_text_block)
    return list_of_blocks

def merge_breaks(breaks, bandwidth = 15):
    main_breaks = []
    for i,each_break in enumerate(breaks):
        if i==0:
            main_breaks.append(each_break)
        else:
            if each_break > breaks[i-1] + bandwidth:
                main_breaks.append(limite)
    return(main_breaks)

def traceLimites():
    pass # utilisée, à copier ?

def draw_page_edges(top_edge, right_edge, bottom_edge, left_edge):
	cv2.rectangle(image, (left_edge,top_edge), (right_edge, bottom_edge), PAGE_EDGES_COLOR, 2)


def find_base_lines(text_block):
    global courbeDensiteZoneV
    left_edge, top_edge, right_edge, bottom_edge = text_block
    left_edge = left_edge + left_page_edge # On repasse dans le référentiel de l'image en rajoutant la coordonnée du bord de la page
    top_edge = top_edge + top_page_edge
    right_edge = right_edge + left_page_edge
    bottom_edge = bottom_edge + top_page_edge
    # définition d'une bande verticale (2e quart) comme échantillon (afin d'éviter à la fois les lettrines, eluminures, et lignes raccourcies)
    block_width = right_edge - left_edge
    sample_width = block_width // 4

    text_block_projection_profile = give_projection_profile(
        gray[top_edge:bottom_edge, left_edge + sample_width:left_edge + sample_width + sample_width], "V")
    avg_brightness_in_text_block = np.average(text_block_projection_profile)
    # print(lumMoyenneZoneTexte)
    # print(f"Courbe densité de la zone de texte ({limiteGauche}, {limiteHaut}, {limiteDroite}, {limiteBas}")
    # print(courbeDensiteZoneV)
    line_tops = []
    line_bottoms = []
    premiereLimiteEstHautDeLigne = None  # Variable qui sert à déterminer si la première limite est bien le haut d'une ligne, et pas un passage obscur->clair du à d'autres zones de l'image
    for i, brightness in enumerate(text_block_projection_profile[:-1]):
        if (brightness >= avg_brightness_in_text_block and avg_brightness_in_text_block > text_block_projection_profile[i + 1]):
            line_tops.append(i)
            # print("Limite Sup : "+str(i+bordHaut+limiteHaut))
            if premiereLimiteEstHautDeLigne == None:
                premiereLimiteEstHautDeLigne = True
        if (brightness <= avg_brightness_in_text_block and avg_brightness_in_text_block < text_block_projection_profile[i + 1]):
            line_bottoms.append(i)
            # print("Limite inf : "+str(i+bordHaut+limiteHaut))
            if premiereLimiteEstHautDeLigne == None:
                premiereLimiteEstHautDeLigne = False
    if premiereLimiteEstHautDeLigne == False:
        line_bottoms.pop(0)
    base_lines = zip(line_tops, line_bottoms)
    return base_lines

class floa_results():
    pass

def analyse(path_to_image_file):
    results = floa_results()
    image = cv2.imread(path_to_image_file)
    image_height, image_width, image_depth = image.shape
    ratio = image_height / image_width
    if image_width >= image_height:  # On réduit la plus petite dimension à 800 px
        new_dimensions = (800, int(ratio * 800))
    else:
        new_dimensions = (int(800 / ratio), 800)
    reduced_image_width = new_dimensions[0]
    reduced_image_height = new_dimensions[1]
    image = cv2.resize(image, new_dimensions)

    floa_results.image_path = path_to_image_file
    floa_results.image = image
    floa_results.image_height = reduced_image_height
    floa_results.image_width = reduced_image_width


    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.GaussianBlur(grey_image, ksize=(21, 21), sigmaX=3,
                            sigmaY=3)  # , sigmaY=1) # ksize de 21 à 43 (forcément impair), sigma de 3 à 5
    floa_results.grey_image = grey_image
    # affiche(gray)
    floa_results.binarized_image = cv2.adaptiveThreshold(grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    return floa_results



