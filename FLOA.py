import numpy as np

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

def bw_thresholding(x, threshold):
    if x <= threshold:
        return 0
    else:
        return 255
def binarize_projection_profile(projection_profile, threshold)#seuilleCourbeDensite(courbeDensite, seuil):
    # lumMoyenne = np.average(courbeDensite)
    # seuil : lumMoyenne = np.median(courbeDensite) # changer le nom si je garde la médiane comme seuil
    binarized_projection_profile = [bw_thresholding(x, threshold) for x in projection_profile]
    return binarized_projection_profile




def smooth_projection_profile(projection_profile, bandwidth = 21) #lisseCourbe(courbe, fenetre = 21):
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

def find_edges_of_page(v_limits, h_limits):
    left_edge = v_limits[1][0]
    right_edge = max([v_limit[0] for v_limit in v_limits])
    top_edge = h_limits[1][0]
    bottom_edge = max([h_limit[0] for h_limit in h_limits])
    return (top_edge, right_edge, bottom_edge, left_edge)

