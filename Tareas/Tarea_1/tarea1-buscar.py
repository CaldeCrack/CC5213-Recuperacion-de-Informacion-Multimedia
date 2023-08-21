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
        matriz_omd = imagen_2.flatten()
        posiciones = numpy.argsort(matriz_omd)
        for i in range(len(posiciones)):
            matriz_omd[posiciones[i]] = i

        # agregar nombre del archivo a la lista de nombres
        lista_nombres.append(nombre)

    #* 2-leer descriptores de R de dir_input_descriptores_R
    archivo_histograma = "{}/{}".format(dir_input_descriptores_R, "descriptor_histograma.npy")
    with open(nombres_R) as f:
        lista_R = f.readlines()
    descriptor_histograma = numpy.load(archivo_histograma)
    #* 3-para cada descriptor q localizar el mas cercano en R
    distancia_histograma = scipy.spatial.distance.cdist(matriz_histograma, descriptor_histograma, metric='cityblock')
    #* 4-escribir en file_output_resultados haciendo print() con el formato:
    numpy.fill_diagonal(distancia_histograma, numpy.inf)
    # más cercanos por histograma
    posiciones_minimas = numpy.argmin(distancia_histograma, axis=1)
    valores_minimos = numpy.amin(distancia_histograma, axis=1)
    resultado_mas_cercanos = []

    for i in range(len(distancia_histograma)):
        query = lista_nombres[i]
        distancia = valores_minimos[i]
        mas_cercano = lista_R[posiciones_minimas[i]]
        resultado_mas_cercanos.append([query, mas_cercano, distancia])
    # más cercanos por OMD
    archivo_omd = "{}/{}".format(dir_input_descriptores_R, "descriptor_omd.npy")
    descriptor_omd = numpy.load(archivo_omd)
    distancia_omd = scipy.spatial.distance.cdist(matriz_omd, descriptor_omd, metric='hamming')

    # output final
    nombres_R = "{}/{}".format(dir_input_descriptores_R, "nombres.data")
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