import numpy as np
import cv2
from matplotlib import pyplot, transforms


BANDWIDTH_FOR_MERGING_BREAKS = 15
PAGE_EDGES_COLOR = (0,255,0)
DEFAULT_BINARIZATION_TRESHOLD = 225 # Threshold for the binarization of the projection profile

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
        if strip_width < max_width_of_black_line and strip_color == 0 and strips[i - 1][3] == "bande blanche" and strips[i + 1][3] == "bande blanche":
            strips[i][3] = "ligne noire"

    return strips

def find_page_edges(v_breaks, h_breaks):
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
        if v_break[3] == "bande noire":
            top_edges.append(v_break[0])
            bottom_edges.append(v_break[1])
            if v_break[0] < max_top_edge:
                max_top_edge = v_break[0]
            if v_break[1] > min_top_edge:
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
                main_breaks.append(each_break)
    return(main_breaks)

def traceLimites():
    pass # utilisée, à copier ?

def draw_page_edges(top_edge, right_edge, bottom_edge, left_edge):
	cv2.rectangle(image, (left_edge,top_edge), (right_edge, bottom_edge), PAGE_EDGES_COLOR, 2)


def find_line_intervals(text_block, results):
    global courbeDensiteZoneV
    left_edge, top_edge, right_edge, bottom_edge = text_block
    left_edge = left_edge + results.left_page_edge # On repasse dans le référentiel de l'image en rajoutant la coordonnée du bord de la page
    top_edge = top_edge + results.top_page_edge
    right_edge = right_edge + results.left_page_edge
    bottom_edge = bottom_edge + results.top_page_edge
    # définition d'une bande verticale (2e quart) comme échantillon (afin d'éviter à la fois les lettrines, eluminures, et lignes raccourcies)
    block_width = right_edge - left_edge
    sample_width = block_width // 4

    text_block_projection_profile = give_projection_profile(results.binarized_image[top_edge:bottom_edge, left_edge + sample_width:left_edge + sample_width + sample_width], "V")
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
    def display(self,mode):
        if mode == "o":
            cv2.imshow(f"Original ({self.image_width}*{self.image_height})", self.image)
        elif mode == "g":
            cv2.imshow(f"Original ({self.image_width}*{self.image_height})", self.grey_image)
        elif mode == "b":
            cv2.imshow(f"Original ({self.image_width}*{self.image_height})", self.binarized_image)
        else:
            return False
        cv2.waitKey(0)
        return (True)
    
    def plot_profile(self, orientation, mode="o"):
        if orientation == "v" and mode == "o":
            data = self.v_projection_profile
            base = pyplot.gca().transData
            rot = transforms.Affine2D().rotate_deg(270)
            pyplot.plot(data, transform = rot + base)
        elif orientation == "h" and mode == "o":
            data = self.h_projection_profile
            pyplot.plot(data)
        elif orientation == "v" and mode == "b":
            data = self.v_binarized_projection_profile
            base = pyplot.gca().transData
            rot = transforms.Affine2D().rotate_deg(270)
            pyplot.plot(data, transform = rot + base)
        elif orientation == "h" and mode == "b":
            data = self.h_binarized_projection_profile
            pyplot.plot(data)
        else:
            return False
        pyplot.show()
        return True
# test.plot_profile("v","o")

    #def __print__:
    #    print self.top_page_edge

def analyse(path_to_image_file):
    results = floa_results()

    # Image preprocessing
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

    results.image_path = path_to_image_file
    results.image = image
    results.image_height = reduced_image_height
    results.image_width = reduced_image_width


    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.GaussianBlur(grey_image, ksize=(21, 21), sigmaX=3,
                            sigmaY=3)  # , sigmaY=1) # ksize de 21 à 43 (forcément impair), sigma de 3 à 5
    results.grey_image = grey_image
    # affiche(gray)
    results.binarized_image = cv2.adaptiveThreshold(grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    # Projection profiles creation
    results.h_projection_profile = give_projection_profile(results.binarized_image, "H")
    results.v_projection_profile = give_projection_profile(results.binarized_image, "V")

    results.h_binarized_projection_profile = binarize_projection_profile(results.h_projection_profile, threshold=DEFAULT_BINARIZATION_TRESHOLD)
    results.v_binarized_projection_profile = binarize_projection_profile(results.v_projection_profile, threshold=DEFAULT_BINARIZATION_TRESHOLD)

    v_breaks = find_breaks(results.h_binarized_projection_profile) #trouveLimites2(courbeDensiteHseuillee)
    h_breaks = find_breaks(results.v_binarized_projection_profile) #trouveLimites2(courbeDensiteVseuillee)

    courbeDensiteZoneV = [] # inutile ?

    # Croping
    y1p, x2p, y2p, x1p = find_page_edges(v_breaks, h_breaks)
    results.top_page_edge = y1p
    results.right_page_edge = x2p
    results.lower_page_edge = y2p
    results.left_page_edge = x1p
    cropped_grey_page = grey_image[y1p:y2p, x1p:x2p]

    # Identification of the blocks of text
    # L'utilisation de lisseCourbe avec une grande fenêtre permet d'ignorer les entrelignes blancs
    # mais ça fait que le cadre est plus grand que le texte sur l'axe vertical
    h_page_projection_profile = give_projection_profile(cropped_grey_page, "H")
    v_page_projection_profile = smooth_projection_profile(give_projection_profile(cropped_grey_page, "V"), 21)

    binarized_h_page_projection_profile = binarize_projection_profile(h_page_projection_profile, threshold=DEFAULT_BINARIZATION_TRESHOLD)
    binarized_v_page_projection_profile = binarize_projection_profile(v_page_projection_profile, threshold=DEFAULT_BINARIZATION_TRESHOLD)

    # USage de h/v pas cohérent avec usage de find_breaks plus haut
    h_breaks_within_page = find_breaks(binarized_h_page_projection_profile)
    v_breaks_within_page = find_breaks(binarized_v_page_projection_profile)

    results.blocks_of_text = find_text_blocks(h_breaks_within_page, v_breaks_within_page)

    # Detection of lines
    # results.lines = []
    # for text_block_number, text_block in enumerate(results.blocks_of_text):
    #     line_intervals = find_line_intervals(text_block, results) #
    #     x1 = text_block[0] + results.left_page_edge
    #     x2 = text_block[2] + results.left_page_edge
    #     for each_line_interval in line_intervals:
    #         y1, y2 = each_line_interval
    #         y = (y1 + y2) // 2 + results.top_page_edge + text_block[1]
    #         results.lines.append(((x1, y), (x2, y), text_block_number))


    return results

test = analyse("/Users/octavejulien/Documents/PIREH/Robocodico/Communication Kalamazoo/Analyses_mss_individuels/Français_1563_jpegs/Fr_1563_220v.jpg")




