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

    # ----- Descriptores -----
    lista_nombres = []
    matriz_histograma = []
    matriz_omd = []
    matriz_intensidad = []
    for nombre in os.listdir(dir_input_imagenes_R):
        if not nombre.endswith(".jpg"):
            continue
        archivo_imagen = "{}/{}".format(dir_input_imagenes_R, nombre)
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

        # agregar nombre del archivo a la lista de nombres
        lista_nombres.append(nombre)

    # guardar info en archivos
    os.makedirs(dir_output_descriptores_R, exist_ok=True)
    archivo_histograma = "{}/{}".format(dir_output_descriptores_R, "descriptor_histograma.npy")
    archivo_omd = "{}/{}".format(dir_output_descriptores_R, "descriptor_omd.npy")
    archivo_intensidad = "{}/{}".format(dir_output_descriptores_R, "descriptor_intensidad.npy")
    nombres_salida = "{}/{}".format(dir_output_descriptores_R, "nombres.data")
    numpy.save(archivo_histograma, matriz_histograma)
    numpy.save(archivo_omd, matriz_omd)
    numpy.save(archivo_intensidad, matriz_intensidad)
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
