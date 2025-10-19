import numpy as np
import cv2
from matplotlib import pyplot, transforms
import os


# TO DO
# Remplacer lower par bottom ?
# Traduire certaines variables dans find_line_intervals
# Résoudre le problème de buffer quand on veut enregistrer le graphique

BANDWIDTH_FOR_MERGING_BREAKS = 15
PAGE_EDGES_COLOR = (0,255,0)
DEFAULT_BINARIZATION_TRESHOLD = 225 # Threshold for the binarization of the projection profile
BLOCKS_COLOR=(0, 0, 255)
LINES_COLOR=(255, 100, 100)
PAGE_COLOR=(0, 255, 0)

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
        return 0
    else:
        return 255

def binarize_projection_profile(projection_profile, threshold): #seuilleCourbeDensite(courbeDensite, seuil):
    # lumMoyenne = np.average(courbeDensite)
    # seuil : lumMoyenne = np.median(courbeDensite) # changer le nom si je garde la médiane comme seuil
    binarized_projection_profile = [bw_thresholding(x, threshold) for x in projection_profile]
    return binarized_projection_profile

def smooth_projection_profile(projection_profile, bandwidth=21): # lisseCourbe(courbe, fenetre = 21):
    length = len(projection_profile)    #longueurCourbe = len(courbe)
    half_bandwidth = bandwidth//2       #demiFenetre = fenetre//2
    start_of_smoothing = -(-bandwidth//2)-1 # debutLissage = -(-fenetre//2)-1
    end_of_smoothing = length - start_of_smoothing #finLissage = longueurCourbe - debutLissage
    smoothed_projection_profile = projection_profile.copy()  #courbeLissee = courbe.copy()
    for i in range(start_of_smoothing, end_of_smoothing):
        smoothed_projection_profile[i] = sum(projection_profile[i-half_bandwidth:i+half_bandwidth+1])//bandwidth # on ne garde que la partie entière pour les performances
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

    text_block_projection_profile = give_projection_profile(results.binarized_image[top_edge:bottom_edge,
                                                            left_edge + sample_width:left_edge +
                                                            sample_width + sample_width], "V")
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
    def __init__(self, path_to_image_file):
        self.file = path_to_image_file
        self.name = os.path.basename(path_to_image_file)

    def __str__(self):
        text_desc = f"""{self.name} 
{self.image_width} × {self.image_height} pix
Page : {self.left_page_edge},{self.top_page_edge}-{self.right_page_edge},{self.lower_page_edge}"""
        nb_lines = self.get_nb_lines()
        for block_number, block in enumerate(self.blocks_of_text):
            x1, y1, x2, y2 = block
            block_desc = f"Block {block_number + 1} : {x1},{y1}-{x2},{y2} ({nb_lines[block_number]} lines)"
            text_desc = text_desc + "\n" + block_desc
        return (text_desc)

    
    def display(self, mode="o", page=True, blocks=True, lines=True, show=True, save_file=None):
        if mode == "o":
            chosen_image = self.original_image.copy()
            title = "Original"
        elif mode == "g":
            chosen_image = cv2.cvtColor(self.grey_image.copy(), cv2.COLOR_GRAY2BGR)
            title = "Greyscale"
        elif mode == "b":
            chosen_image = cv2.cvtColor(self.binarized_image.copy(), cv2.COLOR_GRAY2BGR)
            title = "Binarized"

        if blocks == True:
            for each_block in self.blocks_of_text:
                x1, y1, x2, y2 = each_block
                x1 = x1 + self.left_page_edge
                x2 = x2 + self.left_page_edge
                y1 = y1 + self.top_page_edge
                y2 = y2 + self.top_page_edge
                cv2.rectangle(chosen_image, (x1, y1), (x2, y2), BLOCKS_COLOR, 2)

        if lines == True:
            for each_line in self.lines:
                cv2.line(chosen_image, each_line[0], each_line[1], LINES_COLOR, 1)

        if page == True:
            cv2.rectangle(chosen_image, (self.left_page_edge, self.top_page_edge), (self.right_page_edge, self.lower_page_edge), PAGE_COLOR, 2)

        # if mode == "o":
        #     cv2.imshow(f"Original ({self.image_width}*{self.image_height})", self.image)
        # elif mode == "g":
        #     cv2.imshow(f"Original ({self.image_width}*{self.image_height})", self.grey_image)
        # elif mode == "b":
        #     cv2.imshow(f"Original ({self.image_width}*{self.image_height})", self.binarized_image)
        # #elif mode == "p": # Pour voir uniquement la page seuillée. A corriger pour distinguer le mode et la zone
        #    cv2.imshow(f"Page", self.cropped_binarized_page)
        else:
            return False

        if save_file != None:
            cv2.imwrite(save_file, chosen_image)
        if show == True:
            cv2.imshow(f"{title} ({self.image_width}*{self.image_height})", chosen_image)
            cv2.waitKey(0)
            return (True)


    def plot_profile(self, orientation, mode="o", area="i", show=True, save_file=None): # i : (i)mage, (p)age or (t)ext
        parameters_combination = {
            ("v", "o", "i"): self.v_projection_profile,
            ("h", "o", "i"): self.h_projection_profile,
            ("v", "b", "i"): self.v_binarized_projection_profile,
            ("h", "b", "i"): self.h_binarized_projection_profile,
            ("v", "o", "p"): self.v_page_projection_profile,
            ("h", "o", "p"): self.h_page_projection_profile,
            ("v", "b", "p"): self.binarized_v_page_projection_profile,
            ("h", "b", "p"): self.binarized_h_page_projection_profile
        }

        data = parameters_combination[(orientation, mode, area)]
        if orientation == "h":
            pyplot.plot(data)
        elif orientation == "v":
            base = pyplot.gca().transData
            rot = transforms.Affine2D().rotate_deg(270)
            pyplot.plot(data, transform=rot + base)

        # if orientation == "v" and mode == "o":
        #     data = self.v_projection_profile
        #     base = pyplot.gca().transData
        #     rot = transforms.Affine2D().rotate_deg(270)
        #     pyplot.plot(data, transform = rot + base)
        # elif orientation == "h" and mode == "o":
        #     data = self.h_projection_profile
        #     pyplot.plot(data)
        # elif orientation == "v" and mode == "b":
        #     data = self.v_binarized_projection_profile
        #     base = pyplot.gca().transData
        #     rot = transforms.Affine2D().rotate_deg(270)
        #     pyplot.plot(data, transform = rot + base)
        # elif orientation == "h" and mode == "b":
        #     data = self.h_binarized_projection_profile
        #     pyplot.plot(data)
        else:
            return False
        if show==True:
            pyplot.show()
        if save_file != None:
            pyplot.imsave(save_file, data) # ne marche pas, pb de buffer. Enlever la fonctino save de plot_profile
        return True
# test.plot_profile("v","o")

    def get_page_dim(self, measure="perc"): # measure = [perc]entage ou [pix]el
        pixel_width = self.right_page_edge - self.left_page_edge + 1
        pixel_height = self.lower_page_edge - self.top_page_edge + 1
        if measure == "pix":
            return (pixel_width, pixel_height)
        elif measure == "perc":
            perc_width = round(pixel_width/self.image_width, 4)
            perc_height = round(pixel_height/self.image_height, 4)
            return (perc_width, perc_height)
        else:
            return None

    def get_blocks_dim(self, measure="perc"):
        blocks_dim = []
        for each_block in self.blocks_of_text: # ordre : gauche, haut, droite, bas
            pixel_width = each_block[2] - each_block[0] + 1
            pixel_height = each_block[3] - each_block[1] + 1
            blocks_dim.append((pixel_width,pixel_height))
        if measure == "pix":
            return  blocks_dim
        elif measure == "perc":
            page_width, page_height = self.get_page_dim("pix")
            blocks_dim_perc = []
            for each_block_dim in blocks_dim:
                perc_width = round(each_block_dim[0]/page_width,4)
                perc_height = round(each_block_dim[1]/page_height,4)
                blocks_dim_perc.append((perc_width,perc_height))
            return blocks_dim_perc
        else:
            return None

    def get_nb_blocks(self):
        nb_blocks = len(self.blocks_of_text)
        return nb_blocks

    def get_blocks_coord(self, reference_system = "page"): # fonction peut être superflue avec floa_results.list_of_blocks et fonction de mesure des marges
        blocks_coord = []
        for each_block_coords in self.blocks_of_text:
            formated_block_coords = {}
            if reference_system == "page":
                formated_block_coords["left"] = each_block_coords[0]
                formated_block_coords["top"] = each_block_coords[1]
                formated_block_coords["right"] = each_block_coords[2]
                formated_block_coords["lower"] = each_block_coords[3]
                blocks_coord.append(formated_block_coords)
            elif reference_system == "image":
                formated_block_coords["left"] = each_block_coords[0] + self.left_page_edge
                formated_block_coords["top"] = each_block_coords[1] + self.top_page_edge
                formated_block_coords["right"] = each_block_coords[2] + self.right_page_edge
                formated_block_coords["lower"] = each_block_coords[3] + self.lower_page_edge
                blocks_coord.append(formated_block_coords)
        return blocks_coord

    def get_margins(self, measure="perc"): # Les coordonnées de self.blocks_of_text sont dans le référentiel de la page
        margins_dim = {}
        #"left": None, "top": None, "right": None, "lower": None, "intercolumn" = None
        page_width, page_height = self.get_page_dim(measure="pix")
        margins_dim["top"] = round(self.blocks_of_text[0][1] / page_height,4)
        margins_dim["lower"] = round((page_height - self.blocks_of_text[0][3] +1) / page_height, 4)
        margins_dim["left"] = round(self.blocks_of_text[0][0] / page_width,4)
        margins_dim["right"] = round((page_width - self.blocks_of_text[-1][2] +1) / page_width, 4)

        # Calcul des intercolonnes
        intercolumns_dim = []
        nb_blocks = len(self.blocks_of_text)
        if nb_blocks > 1:
            for i in range(nb_blocks-1): # vérifier index
                block1_right = self.blocks_of_text[i][2]
                block2_left = self.blocks_of_text[i+1][0]
                print(f"gauche : {block1_right} - droite = {block2_left} - distance ={block2_left-block1_right}")
                intercolumn_pix = block2_left - block1_right +1
                intercolumn_perc = round(intercolumn_pix / page_width, 4)
                intercolumns_dim.append(intercolumn_perc)
            margins_dim["intercolumns"] = intercolumns_dim
        return margins_dim





    def get_nb_lines(self):
        nb_blocks = len(self.blocks_of_text)
        nb_lines = []
        for i in range(nb_blocks):
            nb = len([lines for lines in self.lines if lines[2] == i])
            nb_lines.append(nb)
        return nb_lines



def analyse(path_to_image_file):
    results = floa_results(path_to_image_file)

    # Image preprocessing
    original_image = cv2.imread(path_to_image_file)
    image_height, image_width, image_depth = original_image.shape
    ratio = image_height / image_width
    if image_width >= image_height:  # On réduit la plus petite dimension à 800 px
        new_dimensions = (800, int(ratio * 800))
    else:
        new_dimensions = (int(800 / ratio), 800)
    reduced_image_width = new_dimensions[0]
    reduced_image_height = new_dimensions[1]
    original_image = cv2.resize(original_image, new_dimensions)

    results.original_image_path = path_to_image_file
    results.original_image = original_image
    results.image_height = reduced_image_height
    results.image_width = reduced_image_width


    grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.GaussianBlur(grey_image, ksize=(21, 21), sigmaX=3,
                            sigmaY=3)  # , sigmaY=1) # ksize de 21 à 43 (forcément impair), sigma de 3 à 5
    results.grey_image = grey_image
    # affiche(gray)
    results.binarized_image = cv2.adaptiveThreshold(grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
# gray == binarized_image
    # Projection profiles creation
    results.h_projection_profile = give_projection_profile(results.binarized_image, "H")
    results.v_projection_profile = give_projection_profile(results.binarized_image, "V")

    results.h_binarized_projection_profile = binarize_projection_profile(results.h_projection_profile, threshold=DEFAULT_BINARIZATION_TRESHOLD)
    results.v_binarized_projection_profile = binarize_projection_profile(results.v_projection_profile, threshold=DEFAULT_BINARIZATION_TRESHOLD)
# v_binarized_projection_profile == courbeDensiteV
    v_breaks = find_breaks(results.h_binarized_projection_profile) #trouveLimites2(courbeDensiteHseuillee)
    print(v_breaks)
    h_breaks = find_breaks(results.v_binarized_projection_profile) #trouveLimites2(courbeDensiteVseuillee)
# v_breaks == limitesV
    courbeDensiteZoneV = [] # inutile ?

    # Croping
    y1p, x2p, y2p, x1p = find_page_edges(v_breaks, h_breaks)
    # y1p, x2p, y2p, x1p ==  bordHaut, Droit, Bas, Gauche
    print(f"y1p ={y1p}, x2p={x2p}, y2p={y2p}, x1p={x1p}")
    results.top_page_edge = y1p
    results.right_page_edge = x2p
    results.lower_page_edge = y2p
    results.left_page_edge = x1p
    results.cropped_binarized_page = results.binarized_image[y1p:y2p, x1p:x2p]
    # OK : results.cropped_binarized_page == imageGrayRecadree

    # Identification of the blocks of text
    # L'utilisation de lisseCourbe avec une grande fenêtre permet d'ignorer les entrelignes blancs
    # mais ça fait que le cadre est plus grand que le texte sur l'axe vertical
    results.h_page_projection_profile = give_projection_profile(results.cropped_binarized_page, "H")
    results.v_page_projection_profile = smooth_projection_profile(give_projection_profile(results.cropped_binarized_page, "V"), 21)
    # results.v_page_projection_profile ≠ courbeDensitePageV (même si début et fin identique)
    results.binarized_h_page_projection_profile = binarize_projection_profile(results.h_page_projection_profile, threshold=DEFAULT_BINARIZATION_TRESHOLD) # == courbeDensitePageHseuillee
    results.binarized_v_page_projection_profile = binarize_projection_profile(results.v_page_projection_profile, threshold=DEFAULT_BINARIZATION_TRESHOLD) # ≠ courbeDensitePageVseuillee

    # USage de h/v pas cohérent avec usage de find_breaks plus haut
    h_breaks_within_page = find_breaks(results.binarized_h_page_projection_profile)
    v_breaks_within_page = find_breaks(results.binarized_v_page_projection_profile) # resultats différents ici

    results.blocks_of_text = find_text_blocks(h_breaks_within_page, v_breaks_within_page)

    # Detection of lines
    results.lines = []
    for text_block_number, text_block in enumerate(results.blocks_of_text):
        line_intervals = find_line_intervals(text_block, results) #
        x1 = text_block[0] + results.left_page_edge
        x2 = text_block[2] + results.left_page_edge
        for each_line_interval in line_intervals:
            y1, y2 = each_line_interval
            y = (y1 + y2) // 2 + results.top_page_edge + text_block[1]
            results.lines.append(((x1, y), (x2, y), text_block_number))
    # OK


    return results

test = analyse("/Users/octavejulien/Documents/PIREH/Robocodico/Communication Kalamazoo/Analyses_mss_individuels/Français_1563_jpegs/Fr_1563_220v.jpg")
test2 = analyse("/Users/octavejulien/Documents/PIREH/Robocodico/Français_578__btv1b9058898n_108.jpeg")
test3 = analyse("/Users/octavejulien/Documents/PIREH/Robocodico/Page artificielle pour test.jpg")



