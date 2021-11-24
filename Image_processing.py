# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 20:22:13 2021

@author: Miguel
"""

from PIL import Image as IM
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

##############################################################################
# Miguel Angel Solis Orozco
# Innovación, Desarrollo e Investigación II
# Tarea 10
##############################################################################

#Path del archivo .py
path = os.getcwd()

# Función: Genera un nuevo grupo de centroides
# Parámetro:
    # n: número de centroides.
def newcentroids(n):
    new_cents = pd.DataFrame(columns = ['X', 'Y', 'Z'])
    for i in range(n):
        new_cents = new_cents.append({'X': 0, 'Y': 0, 'Z': 0}, ignore_index = True)
    return new_cents

# Función: Leé la imagen en cuestión y la transforma en matriz.
# Parámetro:
    # x: Path y nombre de archivo a partir del path del archivo .py
def tomatrix(x):
    alt_image = np.array(IM.open(path + x))
    return alt_image.shape, np.transpose(alt_image.reshape(-1, 3))

# Función: Leé la matriz en cuestión, la transforma en imagen y la guarda
# en el el path especificado con el nombre especificado.
# Parámetro:
    # x: Path y nombre de archivo a partir del path del archivo .py
    # y: Arreglo a transformar a imagen.
toimage = lambda x, y, z : IM.fromarray(y.reshape(z)).save(path + x)


# Función: Obtiene un arreglo de pixeles en 1D y la cantidad de centroides
# necesarios, procesa la imagen con k-means y devuelve una nueva matroz 1D
# con la imagen procesada.
# Parámetros:
    # image: El arreglo de la imagen.
    # n: Cantidad de centroides.
def kmeans(image, n):
    
    # Definición de centroides iniciales.
    centroids = pd.DataFrame(columns = ['X', 'Y', 'Z'])
    
    # Definición aleatoria de centropides.
    for i in range(n):
        rand = np.random.choice(range(len(np.transpose(image))))
        equis = np.transpose(image)[rand][0]
        ye = np.transpose(image)[rand][1]
        zeta = np.transpose(image)[rand][2]
        equisyezeta = {'X': equis, 'Y': ye, 'Z': zeta}
        centroids = centroids.append(equisyezeta, ignore_index = True)
        
    # Creación de dataframe de valores.
    d = {'X': image[0], 'Y': image[1], 'Z': image[2]}
    valores = pd.DataFrame(d)
    
    # Creación de dataframe de distancias.
    distancias = pd.DataFrame()
    
    new_centroids = newcentroids(n)
    # Inicializamos la columna de asignación en 0
    valores.insert(len(valores.columns), 'Asignación', 0)
    
    # Mientras al menos un centroide se repita...
    while True:
        # Calcular todas las distancias de cada valor a cada centroide.
        for i in range(n):
            dist = []
            for j in range(len(valores)):
                dist.append(np.linalg.norm(np.array((valores['X'][j], valores['Y'][j], valores['Z'][j])) - np.array((centroids['X'][i], centroids['Y'][i], centroids['Z'][i]))))
            distancias[i] = dist

        # Definir el centroide más cercano de cada valor.
        mini = []
        for i in range(len(distancias)):
            m = distancias.loc[i, :].to_list()
            mini.append(m.index(min(m)))
        valores['Asignación'] = mini
        
        # Nuevos centroides
        for i in range(n):
            filtrado = valores[valores['Asignación'] == i]
            new_centroids['X'][i] = np.mean(filtrado['X'])
            new_centroids['Y'][i] = np.mean(filtrado['Y'])
            new_centroids['Z'][i] = np.mean(filtrado['Z'])
            
        # When the centroids get in their place we generate the new image matrix.
        if centroids.equals(new_centroids) == True:
            new_image = []
            for i in range(len(valores)):
                new_image.append(new_centroids.loc[valores['Asignación'][i], :])
            new_image = np.array(new_image)
            return new_image.astype(np.uint8)
        
        centroids = new_centroids
        new_centroids = newcentroids(n)
        
# Función: Obtiene un arreglo de pixeles en 1D y la cantidad de centroides
# necesarios, procesa la imagen con kohonen y devuelve una nueva matroz 1D
# con la imagen procesada.
# Parámetros:
    # image: El arreglo de la imagen.
    # n: Cantidad de centroides.
def kohonen(image, n):
    
    # Definición de centroides iniciales.
    centroids = pd.DataFrame(columns = ['X', 'Y', 'Z'])

    # Definición aleatoria de centropides.
    for i in range(n):
        rand = np.random.choice(range(len(np.transpose(image))))
        equis = np.transpose(image)[rand][0]
        ye = np.transpose(image)[rand][1]
        zeta = np.transpose(image)[rand][2]
        equisyezeta = {'X': equis, 'Y': ye, 'Z': zeta}
        centroids = centroids.append(equisyezeta, ignore_index = True)
        
    # Creación de dataframe de valores.
    d = {'X': image[0], 'Y': image[1], 'Z': image[2]}
    valores = pd.DataFrame(d)
    
    # Creación de dataframe de distancias.
    distancias = pd.DataFrame()
    
    new_centroids = newcentroids(n)
    # Inicializamos la columna de asignación en 0
    valores.insert(len(valores.columns), 'Asignación', 0)
    
    # Mientras al menos un centroide se repita...
    while True:
        # Calcular todas las distancias de cada valor a cada centroide.
        for i in range(n):
            dist = []
            for j in range(len(valores)):
                dist.append(np.linalg.norm(np.array((valores['X'][j], valores['Y'][j], valores['Z'][j])) - np.array((centroids['X'][i], centroids['Y'][i], centroids['Z'][i]))))
            distancias[i] = dist

        # Definir el centroide más cercano de cada valor.
        mini = []
        for i in range(len(distancias)):
            m = distancias.loc[i, :].to_list()
            mini.append(m.index(min(m)))
        valores['Asignación'] = mini
        
        # Carrusel de Kohonen
        for i in range(len(valores)):
            paso = 2 # Paso
            cen = np.atleast_2d(new_centroids.loc[valores['Asignación'][i]].to_numpy())
            dist = [valores['X'][i], valores['Y'][i], valores['Z'][i]]
            new_centroids.loc[valores['Asignación'][i]] = cen + (1/paso) * (dist - cen)
   
        # When the centroids get in their place we generate the new image matrix.
        if centroids.equals(new_centroids) == True:
            new_image = []
            for i in range(len(valores)):
                new_image.append(new_centroids.loc[valores['Asignación'][i], :])
            new_image = np.array(new_image)
            return new_image.astype(np.uint8)
        
        centroids = new_centroids
        new_centroids = newcentroids(n)


# K-means

shape, matrix = tomatrix('/tarea10/imagen1Deutschland-1.jpg')
result = kmeans(matrix, 2)
toimage('/tarea10/km-alemania2.jpg', result, shape)
result = kmeans(matrix, 3)
toimage('/tarea10/km-alemania3.jpg', result, shape)
result = kmeans(matrix, 10)
toimage('/tarea10/km-alemania10.jpg', result, shape)

shape, matrix = tomatrix('/tarea10/imagen2México.jpg')
result = kmeans(matrix, 2)
toimage('/tarea10/km-mexico2.jpg', result, shape)
result = kmeans(matrix, 3)
toimage('/tarea10/km-mexico3.jpg', result, shape)
result = kmeans(matrix, 10)
toimage('/tarea10/km-mexico10.jpg', result, shape)

shape, matrix = tomatrix('/tarea10/imagen3Gandhi.jpg')
result = kmeans(matrix, 2)
toimage('/tarea10/km-gandhi2.jpg', result, shape)
result = kmeans(matrix, 3)
toimage('/tarea10/km-gandhi3.jpg', result, shape)
result = kmeans(matrix, 10)
toimage('/tarea10/km-gandhi10.jpg', result, shape)

shape, matrix = tomatrix('/tarea10/imagen4Lauterbrunnen.jpg')
result = kmeans(matrix, 2)
toimage('/tarea10/km-paisajesuizo2.jpg', result, shape)
result = kmeans(matrix, 3)
toimage('/tarea10/km-paisajesuizo3.jpg', result, shape)
result = kmeans(matrix, 10)
toimage('/tarea10/km-paisajesuizo10.jpg', result, shape)


# Kohonen

shape, matrix = tomatrix('/tarea10/imagen1Deutschland-1.jpg')
result = kohonen(matrix, 2)
toimage('/tarea10/knn-alemania2.jpg', result, shape)
result = kohonen(matrix, 3)
toimage('/tarea10/knn-alemania3.jpg', result, shape)
result = kohonen(matrix, 10)
toimage('/tarea10/knn-alemania10.jpg', result, shape)

shape, matrix = tomatrix('/tarea10/imagen2México.jpg')
result = kohonen(matrix, 2)
toimage('/tarea10/knn-mexico2.jpg', result, shape)
result = kohonen(matrix, 3)
toimage('/tarea10/knn-mexico3.jpg', result, shape)
result = kohonen(matrix, 10)
toimage('/tarea10/knn-mexico10.jpg', result, shape)

shape, matrix = tomatrix('/tarea10/imagen3Gandhi.jpg')
result = kohonen(matrix, 2)
toimage('/tarea10/knn-gandhi2.jpg', result, shape)
result = kohonen(matrix, 3)
toimage('/tarea10/knn-gandhi3.jpg', result, shape)
result = kohonen(matrix, 10)
toimage('/tarea10/knn-gandhi10.jpg', result, shape)

shape, matrix = tomatrix('/tarea10/imagen4Lauterbrunnen.jpg')
result = kohonen(matrix, 2)
toimage('/tarea10/knn-paisajesuizo2.jpg', result, shape)
result = kohonen(matrix, 3)
toimage('/tarea10/knn-paisajesuizo3.jpg', result, shape)
result = kohonen(matrix, 10)
toimage('/tarea10/knn-paisajesuizo10.jpg', result, shape)
