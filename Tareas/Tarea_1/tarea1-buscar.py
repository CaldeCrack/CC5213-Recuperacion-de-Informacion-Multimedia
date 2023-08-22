# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 15 de agosto de 2023
# Alumno: Andrés Calderón Guardia

import sys, os.path, numpy, cv2, scipy

def tarea1_buscar(dir_input_imagenes_Q, dir_input_descriptores_R, file_output_resultados):
    if not os.path.isdir(dir_input_imagenes_Q):
        print("ERROR: no existe directorio {}".format(dir_input_imagenes_Q))
        sys.exit(1)
    elif not os.path.isdir(dir_input_descriptores_R):
        print("ERROR: no existe directorio {}".format(dir_input_descriptores_R))
        sys.exit(1)
    elif os.path.exists(file_output_resultados):
        print("ERROR: ya existe archivo {}".format(file_output_resultados))
        sys.exit(1)

    # ----- Descriptores -----
    lista_nombres = []
    matriz_histograma = []
    matriz_omd = []
    matriz_intensidad = []
    matriz_eh = []
    for nombre in os.listdir(dir_input_imagenes_Q):
        if not nombre.endswith(".jpg"):
            continue
        archivo_imagen = "{}/{}".format(dir_input_imagenes_Q, nombre)
        # ----- Histograma por Zonas -----
        # divisiones
        num_zonas_x = 1
        num_zonas_y = 16
        num_bins_por_zona = 16
        ecualizar = False
        # leer imagen
        imagen = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
        if ecualizar:
            imagen = cv2.equalizeHist(imagen)
        # procesar cada zona
        descriptor = []
        for j in range(num_zonas_y):
            desde_y = int(imagen.shape[0] / num_zonas_y * j)
            hasta_y = int(imagen.shape[0] / num_zonas_y * (j+1))
            for i in range(num_zonas_x):
                desde_x = int(imagen.shape[1] / num_zonas_x * i)
                hasta_x = int(imagen.shape[1] / num_zonas_x * (i+1))
                # recortar zona de la imagen
                zona = imagen[desde_y : hasta_y, desde_x : hasta_x]
                # histograma de los pixeles de la zona
                histograma, limites = numpy.histogram(zona, bins=num_bins_por_zona, range=(0,255))
                # normalizar histograma (bins suman 1)
                histograma = histograma / numpy.sum(histograma)
                # agregar descriptor de la zona al descriptor global
                descriptor.extend(histograma)
        # agregar descriptor a la matriz de descriptores
        if len(matriz_histograma) == 0:
            matriz_histograma = descriptor
        else:
            matriz_histograma = numpy.vstack([matriz_histograma, descriptor])

        # ----- OMD -----
        imagen_1 = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
        imagen_2 = cv2.resize(imagen_1, (4, 4), interpolation=cv2.INTER_AREA)
        descriptor = imagen_2.flatten()
        posiciones = numpy.argsort(descriptor)
        for i in range(len(posiciones)):
            descriptor[posiciones[i]] = i
        if len(matriz_omd) == 0:
            matriz_omd = descriptor
        else:
            matriz_omd = numpy.vstack([matriz_omd, descriptor])

        # ----- Vector de Intensidades -----
        imagen_1 = cv2.imread(archivo_imagen, cv2.IMREAD_COLOR)
        imagen_2 = cv2.resize(imagen_1, (20, 20), interpolation=cv2.INTER_AREA)
        # flatten convierte una matriz de nxm en un array de largo nxm
        descriptor = imagen_2.flatten()
        if len(matriz_intensidad) == 0:
            matriz_intensidad = descriptor
        else:
            matriz_intensidad = numpy.vstack([matriz_intensidad, descriptor])

        # ----- EH -----
        imagen_1 = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
        imagen_1 = cv2.equalizeHist(imagen_1)
        imagen_1 = cv2.resize(imagen_1, (10, 10), interpolation=cv2.INTER_AREA)
        descriptor = imagen_1.flatten()
        if len(matriz_eh) == 0:
            matriz_eh = descriptor
        else:
            matriz_eh = numpy.vstack([matriz_eh, descriptor])

        # agregar nombre del archivo a la lista de nombres
        lista_nombres.append(nombre)

    # nombre de los archivos
    nombres_R = "{}/{}".format(dir_input_descriptores_R, "nombres.data")
    with open(nombres_R) as f:
        lista_R = f.readlines()

    # más cercanos por Histograma
    archivo_histograma = "{}/{}".format(dir_input_descriptores_R, "descriptor_histograma.npy")
    descriptor_histograma = numpy.load(archivo_histograma)
    distancia_histograma = scipy.spatial.distance.cdist(matriz_histograma, descriptor_histograma, metric='cityblock')
    # normalización
    distancia_histograma /= numpy.max(distancia_histograma)

    # más cercanos por OMD
    archivo_omd = "{}/{}".format(dir_input_descriptores_R, "descriptor_omd.npy")
    descriptor_omd = numpy.load(archivo_omd)
    distancia_omd = scipy.spatial.distance.cdist(matriz_omd, descriptor_omd, metric='hamming')
    # normalización
    distancia_omd /= numpy.mean(distancia_omd)

    # más cercanos por Vector de Intensidades
    archivo_intensidad = "{}/{}".format(dir_input_descriptores_R, "descriptor_intensidad.npy")
    descriptor_intensidad = numpy.load(archivo_intensidad)
    distancia_intensidad = scipy.spatial.distance.cdist(matriz_intensidad, descriptor_intensidad, metric='cityblock')
    # normalización
    distancia_intensidad /= numpy.mean(distancia_intensidad)

    # más cercanos por histograma ecualizado
    archivo_eh = "{}/{}".format(dir_input_descriptores_R, "descriptor_eh.npy")
    descriptor_eh = numpy.load(archivo_eh)
    distancia_eh = scipy.spatial.distance.cdist(matriz_eh, descriptor_eh, metric='euclidean')
    # normalización
    distancia_eh /= numpy.mean(distancia_eh)

    # más cercanos
    min_matrix = numpy.minimum(distancia_histograma, distancia_omd)
    min_matrix2 = numpy.minimum(distancia_eh, distancia_intensidad)
    min_matrix = numpy.minimum(min_matrix, min_matrix2)
    posiciones_minimas = numpy.argmin(min_matrix, axis=1)
    valores_minimos = numpy.amin(min_matrix, axis=1)
    resultado_mas_cercanos = []
    for i in range(len(min_matrix)):
        query = lista_nombres[i]
        distancia = valores_minimos[i]
        mas_cercano = lista_R[posiciones_minimas[i]]
        resultado_mas_cercanos.append([query, mas_cercano, distancia])

    # output final
    with open(file_output_resultados, 'w') as f:
        for elem in resultado_mas_cercanos:
            print("{}\t{}\t{}".format(elem[0], elem[1][0:-1], elem[2]), file=f)

# inicio de la tarea
if len(sys.argv) < 4:
    print("Uso: {} [dir_input_imagenes_Q] [dir_input_descriptores_R] [file_output_resultados]".format(sys.argv[0]))
    sys.exit(1)

dir_input_imagenes_Q = sys.argv[1]
dir_input_descriptores_R = sys.argv[2]
file_output_resultados = sys.argv[3]

tarea1_buscar(dir_input_imagenes_Q, dir_input_descriptores_R, file_output_resultados)