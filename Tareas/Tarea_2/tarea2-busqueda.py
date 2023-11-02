# CC5213 - TAREA 2
# 28 de septiembre de 2023
# Alumno: Andrés Calderón Guardia

import sys, os.path, numpy, scipy

def tarea2_busqueda(dir_descriptores_q, dir_descriptores_r, dir_resultados_knn):
    if not os.path.isdir(dir_descriptores_q):
        print("ERROR: no existe directorio {}".format(dir_descriptores_q))
        sys.exit(1)
    elif not os.path.isdir(dir_descriptores_r):
        print("ERROR: no existe directorio {}".format(dir_descriptores_r))
        sys.exit(1)
    elif os.path.exists(dir_resultados_knn):
        print("ERROR: ya existe archivo {}".format(dir_resultados_knn))
        sys.exit(1)
    archivo_canciones = "{}/{}".format(dir_descriptores_r, "descriptores_canciones.npy")
    descriptores_canciones = numpy.load(archivo_canciones)
    archivo_radio = "{}/{}".format(dir_descriptores_q, "descriptores_radio.npy")
    descriptores_radio = numpy.load(archivo_radio)
    matriz_distancias = scipy.spatial.distance.cdist(descriptores_radio, descriptores_canciones, metric='cityblock')

    # nombres de los archivos
    nombres_canciones = "{}/{}".format(dir_descriptores_r, "timestamps_canciones.data")
    with open(nombres_canciones) as f:
        lista_canciones = f.readlines()
    nombres_radio = "{}/{}".format(dir_descriptores_q, "timestamps_radio.data")
    with open(nombres_radio) as f:
        lista_radio = f.readlines()

    # obtener la posicion del más cercano por fila
    posicion_min = numpy.argmin(matriz_distancias, axis=1)
    minimo = numpy.amin(matriz_distancias, axis=1)

    os.makedirs(dir_resultados_knn, exist_ok=True)
    # guardar resultados en un archivo
    archivo_resultados = "{}/{}".format(dir_resultados_knn, "knn.data")
    with open(archivo_resultados, 'w') as f:
        for i in range(len(lista_radio)):
            query = lista_radio[i]
            cancion = lista_canciones[posicion_min[i]]
            distancia = minimo[i]
            print("{}\t{}\t{:.2f}".format(query[:-1], cancion[:-1], distancia), file=f)

# inicio de la tarea
if len(sys.argv) < 4:
    print("Uso: {} [dir_descriptores_q] [dir_descriptores_r] [dir_resultados_knn]".format(sys.argv[0]))
    sys.exit(1)

dir_descriptores_q = sys.argv[1]
dir_descriptores_r = sys.argv[2]
dir_resultados_knn = sys.argv[3]

tarea2_busqueda(dir_descriptores_q, dir_descriptores_r, dir_resultados_knn)