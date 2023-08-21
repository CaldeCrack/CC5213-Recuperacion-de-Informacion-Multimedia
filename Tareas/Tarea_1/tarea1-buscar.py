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
    # Implementar la tarea:
    #* 1-leer imágenes en dir_input_imagenes_Q y calcular descriptores cada imagen
    for nombre in os.listdir(dir_input_imagenes_Q):
        if not nombre.endswith(".jpg"):
            continue
        archivo_imagen = "{}/{}".format(dir_input_imagenes_Q, nombre)
        # divisiones
        num_zonas_x = 2
        num_zonas_y = 2 
        num_bins_por_zona = 8
        ecualizar = True
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
        if len(matriz_descriptores) == 0:
            matriz_descriptores = descriptor
        else:
            matriz_descriptores = numpy.vstack([matriz_descriptores, descriptor])
    
    #* 2-leer descriptores de R de dir_input_descriptores_R
    descriptores_R = []
    #* 3-para cada descriptor q localizar el mas cercano en R
    matriz_distancias = scipy.spatial.distance.cdist(matriz_descriptores, matriz_descriptores, metric='cityblock')
    #* 4-escribir en file_output_resultados haciendo print() con el formato: 
    with open(file_output_resultados, 'w') as f:
        pass
        # for ...
        #     print("{}\t{}\t{}".format(imagen_q, imagen_r, distancia), file=f)


# inicio de la tarea
if len(sys.argv) < 4:
    print("Uso: {} [dir_input_imagenes_Q] [dir_input_descriptores_R] [file_output_resultados]".format(sys.argv[0]))
    sys.exit(1)

dir_input_imagenes_Q = sys.argv[1]
dir_input_descriptores_R = sys.argv[2]
file_output_resultados = sys.argv[3]

tarea1_buscar(dir_input_imagenes_Q, dir_input_descriptores_R, file_output_resultados)
