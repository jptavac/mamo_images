
import numpy as np
import pandas as pd
from PIL import Image
import cv2 
import os
from tqdm import tqdm

import mammo_processing as mp

################################################################################

def load_DDSM_images (DDSM_path, DDSM_df_route, indexes=0, set_indexes=False):
    # Esta funcion solo sirve para mi caso particular, no es generalizable.
    # Carga solo aquellas imagenes de vista MLO que tengan una sola mascara asignada (un solo contorno).
    # Esto es para simplificar el problema de deteccion a un solo contorno en lugar de multiples.
    # Devuelve una lista con las imagenes cargadas y otra con las mascaras.
    # Indexes indica los indices del dataframe a revisar, si es que solo se quiere revisar una parte del dataset.
    # Para utilizar indexes debe ponerse set_indexes a TRUE.
    # Por defecto solo se revisan los primeros 10 elementos del dataset.
    DDSM_df = pd.read_excel(DDSM_df_route)
    # Las columnas 'fullPath', 'Tumour_Contour' y 'Tumout_Contour2' del dataframe original contiene '\' en lugar de '/', por lo que debo reemplazarlos al tratar de cargar la imagen. Modifico el dataframe con esta consideracion.
    print('Comenzando adaptacion del dataframe.')
    for index in tqdm(range(0,len(DDSM_df))):
        DDSM_df['fullPath'       ][index] = DDSM_df['fullPath'       ][index].replace('\\' , '/')
        DDSM_df['Tumour_Contour' ][index] = DDSM_df['Tumour_Contour' ][index].replace('\\' , '/')
        DDSM_df['Tumour_Contour2'][index] = DDSM_df['Tumour_Contour2'][index].replace('\\' , '/')
    image_list = []     #creo una lista de imagenes vacia
    mask_list  = []     #creo una lista de mascaras vacia
    image_info = pd.DataFrame(columns=DDSM_df.columns) #creo un dataframe con las mismas columnas que el DDSM original
    if set_indexes==False:
        indexes = range(0, len(DDSM_df))
    print('Comenzando carga de las imagenes.')
    for index in tqdm(indexes):              #recorro determinada cantidad de lineas en DDSM_df y copio la informacion que pase los filtros
        if (DDSM_df['View'][index]=='MLO'):
            if (DDSM_df['Tumour_Contour2'][index]=='-'):
                #agrego la imagen como arreglo a la lista
                image_list.append(np.array(Image.open(DDSM_path + DDSM_df['fullPath'][index]))) 
                #copio la informacion de la imagen al dataframe filtrado, notar el .loc[[]] con doble corchete para que devuelva un dataframe
                #que pueda ser concatenado
                image_info = pd.concat([image_info, DDSM_df.iloc[[index]]], ignore_index=True)   
                #si no existe mascara para la imagen genero una totalmente negra para agregar a la lista de mascaras
                if (DDSM_df['Tumour_Contour'][index]=='-'):
                    #como la imagen base es la ultima de la lista, tomo la logitud de la lista -1 para acceder a la ultima imagen cargada
                    mask_list.append(mp.array_generate_black_background(image_list[len(image_list)-1]))
                #si existe mascara la cargo y la binarizo
                else:
                    mask_list.append(mp.array_binarize_with_threshold(np.array(Image.open(DDSM_path + DDSM_df['Tumour_Contour'][index])),20)) #agrego la mascara como arreglo a la lista    
    return image_list, mask_list, image_info

################################################################################

def load_and_resize_DDSM_images (DDSM_path, DDSM_df_route, indexes=0, set_indexes=False, new_size=(512,512), expand_dimensions=False, return_as_array=False, edit_dataset=False):
    # Esta funcion solo sirve para mi caso particular, no es generalizable.
    # Carga solo aquellas imagenes de vista MLO que tengan una sola mascara asignada (un solo contorno).
    # Esto es para simplificar el problema de deteccion a un solo contorno en lugar de multiples.
    # Devuelve una lista con las imagenes cargadas y otra con las mascaras.
    # Indexes indica los indices del dataframe a revisar, si es que solo se quiere revisar una parte del dataset.
    # Para utilizar indexes debe ponerse set_indexes a TRUE.
    # Por defecto solo se revisan los primeros 10 elementos del dataset.
    DDSM_df = pd.read_excel(DDSM_df_route)
    # Las columnas 'fullPath', 'Tumour_Contour' y 'Tumout_Contour2' del dataframe original contiene '\' en lugar de '/', por lo que debo reemplazarlos al tratar de cargar la imagen. Modifico el dataframe con esta consideracion.
    if edit_dataset==True:
        print('Comenzando adaptacion del dataframe.')
        for index in range(0,len(DDSM_df)):
            DDSM_df['fullPath'       ][index] = DDSM_df['fullPath'       ][index].replace('\\' , '/')
            DDSM_df['Tumour_Contour' ][index] = DDSM_df['Tumour_Contour' ][index].replace('\\' , '/')
            DDSM_df['Tumour_Contour2'][index] = DDSM_df['Tumour_Contour2'][index].replace('\\' , '/')
    image_list = []     #creo una lista de imagenes vacia
    mask_list  = []     #creo una lista de mascaras vacia
    image_info = pd.DataFrame(columns=DDSM_df.columns) #creo un dataframe con las mismas columnas que el DDSM original
    if set_indexes==False:
        indexes = range(0, len(DDSM_df))
    print('Comenzando carga de las imagenes.')
    for index in tqdm(indexes):              #recorro determinada cantidad de lineas en DDSM_df y copio la informacion que pase los filtros
        if (DDSM_df['View'][index]=='MLO'):
            if (DDSM_df['Tumour_Contour2'][index]=='-'):
                #agrego la imagen como arreglo a la lista
                image_list.append(np.array(Image.open(DDSM_path + DDSM_df['fullPath'][index]).resize(new_size))) 
                #copio la informacion de la imagen al dataframe filtrado, notar el .loc[[]] con doble corchete para que devuelva un dataframe
                #que pueda ser concatenado
                image_info = pd.concat([image_info, DDSM_df.iloc[[index]]], ignore_index=True)   
                #si no existe mascara para la imagen genero una totalmente negra para agregar a la lista de mascaras
                if (DDSM_df['Tumour_Contour'][index]=='-'):
                    #como la imagen base es la ultima de la lista, tomo la logitud de la lista -1 para acceder a la ultima imagen cargada
                    mask_list.append(mp.array_generate_black_background(image_list[len(image_list)-1]))
                #si existe mascara la cargo y la binarizo
                else:
                    mask_list.append(mp.array_binarize_with_threshold(np.array(Image.open(DDSM_path + DDSM_df['Tumour_Contour'][index]).resize(new_size)),20)) #agrego la mascara como arreglo a la lista    
    
    if expand_dimensions==True:
        for index in range(len(image_list)):
            image_list[index] = np.expand_dims(image_list[index], 2)
            mask_list [index] = np.expand_dims(mask_list [index], 2)
    
    if return_as_array==True:
      return np.array(image_list), np.array(mask_list), image_info
    else:
        return image_list, mask_list, image_info

################################################################################

def save_DDSM_images (image_list, mask_list, image_info_df, DDSM_save_path, DDSM_df_save_route, indexes = 0, set_indexes=False):
    # Esta funcion solo sirve para mi caso particular, no es generalizable.
    # Devuelve una lista con las imagenes cargadas y otra con las mascaras.
    # Indexes indica los indices del dataframe a revisar, para no tener que recorrer la totalidad. 
    # Para utilizar indexes debe ponerse set_indexes a TRUE.
    image_info_df.to_excel(DDSM_df_save_route)

    if set_indexes==False:
        indexes = range(0, len(image_info_df))
    print('Comenzando guardado de imagenes.')
    for index in tqdm(indexes):              #recorro determinada cantidad de lineas en DDSM_df y copio la informacion que pase los filtros.
        full_save_path = DDSM_save_path + image_info_df['fullPath'][index]   # Genero el path completo con el nombre de la imagen.
        # A continuacion compruebo si el path para guardar la imagen existe.
        # Todos los paths cargados en el dataframe incluyen el nombre de la imagen.
        # Debo generar el path sin el nombre de la imagen.
        # El nombre de la imagen siempre tiene 21 caracteres para mamografia izquierda, 22 para derecha.
        # Para eliminar el nombre de la imagen debo exceptuar estos ultimos caracteres segun corresponda.
        # Para hacer esto tomo el "full_save_path" y reviso si tiene "RIGHT" o "LEFT" en su nombre.
        # En base a eso decido si debo eliminar los ultimos 21 o 22 caracteres.
        if 'LEFT' in full_save_path:
            erase_from_path = 21
        else:
            erase_from_path = 22
            
        if os.path.exists(full_save_path[0 : len(full_save_path)-erase_from_path])==False :              # Reviso si el path existe.
            os.makedirs(full_save_path[0 : len(full_save_path)-erase_from_path])                            # Si no existe, lo creo.
        cv2.imwrite(full_save_path, image_list[index])          # Guardo la imagen.
            
        # El nombre de la mascara (cuando esta existe) es el mismo de la imagen con el sufijo "_mask".
        # En lugar de comprobar si existe o no una mascara, genero el path completo en base al path
        # de imagen.
        full_mask_save_path  = full_save_path.replace('.jpg', '_Mask.jpg') 
        cv2.imwrite(full_mask_save_path, mask_list[index])      # Guardo la mascara.


def load_images_for_unet(DDSM_path, DDSM_df_route, indexes=0, set_indexes=False, new_size=(512,512)):
    image_list, mask_list, ddsm_df = load_and_resize_DDSM_images(DDSM_path, DDSM_df_route, indexes=indexes, set_indexes=set_indexes, new_size=new_size)
    for index in range(len(image_list)):
        image_list[index] = np.expand_dims(image_list[index], 2)
        mask_list [index] = np.expand_dims(mask_list [index], 2)
    return np.array(image_list), np.array(mask_list)

################################################################################

def load_processed_DDSM_images_and_maps (DDSM_path, DDSM_df_route, indexes=0, set_indexes=False, load_maps = False, expand_dimensions=False, return_as_array=False):
    
    DDSM_df = pd.read_excel(DDSM_df_route)

    if load_maps:
        image_list   = []     #creo una lista de imagenes vacia
        mask_list    = []     #creo una lista de mascaras vacia
        fracmap_list = []
        lacmap_list  = []
        image_info = pd.DataFrame(columns=DDSM_df.columns) #creo un dataframe con las mismas columnas que el DDSM original

        if set_indexes==False:
            indexes = range(0, len(DDSM_df))
        print('Comenzando carga de las imagenes.')
        for index in tqdm(indexes):              #recorro determinada cantidad de lineas en DDSM_df y copio la informacion que pase los filtros
            if (DDSM_df['View'][index]=='MLO'):
                if (DDSM_df['Tumour_Contour2'][index]=='-'):
                    #agrego la imagen como arreglo a la lista
                    image_path   = DDSM_path + DDSM_df['fullPath'][index]
                    mask_path    = image_path.replace('.jpg', '_Mask.jpg') 
                    fracmap_path = image_path.replace('.jpg', '_fractal_map.jpg') 
                    lacmap_path  = image_path.replace('.jpg', '_lacunarity_map.jpg') 

                    image_list.append(np.array(Image.open(image_path)))
                    mask_list.append(np.array(Image.open(mask_path)))
                    fracmap_list.append(np.array(Image.open(fracmap_path)))
                    lacmap_list.append(np.array(Image.open(lacmap_path)))
                    
                    #copio la informacion de la imagen al dataframe filtrado, notar el .loc[[]] con doble corchete para que devuelva un dataframe
                    #que pueda ser concatenado
                    image_info = pd.concat([image_info, DDSM_df.iloc[[index]]], ignore_index=True)     
        
        if expand_dimensions==True:
            for index in range(len(image_list)):
                image_list[index] = np.expand_dims(image_list[index], 2)
                mask_list [index] = np.expand_dims(mask_list [index], 2)
                fracmap_list [index] = np.expand_dims(fracmap_list [index], 2)
                lacmap_list [index] = np.expand_dims(lacmap_list [index], 2)
        
        if return_as_array==True:
            return np.array(image_list), np.array(mask_list), np.array(fracmap_list), np.array(lacmap_list), image_info
        else:
            return image_list, mask_list, fracmap_list, lacmap_list, image_info
    
    if not load_maps:
        image_list   = []     #creo una lista de imagenes vacia
        mask_list    = []     #creo una lista de mascaras vacia
        image_info = pd.DataFrame(columns=DDSM_df.columns) #creo un dataframe con las mismas columnas que el DDSM original

        if set_indexes==False:
            indexes = range(0, len(DDSM_df))
        print('Comenzando carga de las imagenes.')
        for index in tqdm(indexes):              #recorro determinada cantidad de lineas en DDSM_df y copio la informacion que pase los filtros
            if (DDSM_df['View'][index]=='MLO'):
                if (DDSM_df['Tumour_Contour2'][index]=='-'):
                    #agrego la imagen como arreglo a la lista
                    image_path   = DDSM_path + DDSM_df['fullPath'][index]
                    mask_path    = image_path.replace('.jpg', '_Mask.jpg') 
                    
                    image_list.append(np.array(Image.open(image_path)))
                    mask_list.append(np.array(Image.open(mask_path)))
                    
                    #copio la informacion de la imagen al dataframe filtrado, notar el .loc[[]] con doble corchete para que devuelva un dataframe
                    #que pueda ser concatenado
                    image_info = pd.concat([image_info, DDSM_df.iloc[[index]]], ignore_index=True)     
        
        if expand_dimensions==True:
            for index in range(len(image_list)):
                image_list[index] = np.expand_dims(image_list[index], 2)
                mask_list [index] = np.expand_dims(mask_list [index], 2)
                
        if return_as_array==True:
            return np.array(image_list), np.array(mask_list), image_info
        else:
            return image_list, mask_list, image_info


################################################################################

def load_processed_images_and_maps_in_three_channels(DDSM_path, DDSM_df_route, indexes=0, set_indexes=False):
    images, masks, fracmaps, lacmaps, info_df = load_processed_DDSM_images_and_maps (DDSM_path, DDSM_df_route, indexes, set_indexes, expand_dimensions=True, return_as_array=True)
    
    tri_channel_input = np.concatenate((images, fracmaps, lacmaps), axis=3)
    tri_channel_masks = np.concatenate((masks , masks   , masks  ), axis=3)

    return tri_channel_input, tri_channel_masks
