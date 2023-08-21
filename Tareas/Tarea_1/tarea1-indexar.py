# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 15 de agosto de 2023
# Alumno: Andrés Calderón Guardia

import sys, os.path, numpy, cv2

def tarea1_indexar(dir_input_imagenes_R, dir_output_descriptores_R):
    if not os.path.isdir(dir_input_imagenes_R):
        print("ERROR: no existe directorio {}".format(dir_input_imagenes_R))
        sys.exit(1)
    elif os.path.exists(dir_output_descriptores_R):
        print("ERROR: ya existe directorio {}".format(dir_output_descriptores_R))
        sys.exit(1)
    #* 1-leer imágenes en dir_input_imagenes_R y calcular descriptores cada imagen
    lista_nombres = []
    matriz_descriptores = []
    for nombre in os.listdir(dir_input_imagenes_R):
        if not nombre.endswith(".jpg"):
            continue
        archivo_imagen = "{}/{}".format(dir_input_imagenes_R, nombre)
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
        # agregar nombre del archivo a la lista de nombres
        lista_nombres.append(nombre)

    #* 3-escribir en dir_output_descriptores_R los descriptores calculados (crear uno o más archivos)
    os.makedirs(dir_output_descriptores_R, exist_ok=True)
    archivo_salida = "{}/{}".format(dir_output_descriptores_R, "descriptores.npy")
    nombres_salida = "{}/{}".format(dir_output_descriptores_R, "nombres.data")
    numpy.save(archivo_salida, matriz_descriptores)
    with open(nombres_salida, "w") as f:
        for i in range(len(lista_nombres)):
            f.write(str(lista_nombres[i]) + "\n")

# inicio de la tarea
if len(sys.argv) < 3:
    print("Uso: {} [dir_input_imagenes_R] [dir_output_descriptores_R]".format(sys.argv[0]))
    sys.exit(1)

dir_input_imagenes_R = sys.argv[1]
dir_output_descriptores_R = sys.argv[2]

tarea1_indexar(dir_input_imagenes_R, dir_output_descriptores_R)
