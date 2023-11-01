# CC5213 - TAREA 2
# 28 de septiembre de 2023
# Alumno: [nombre]

import sys
import os.path


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
    # Implementar la busqueda
    #  1-leer descriptores de Q de dir_descriptores_q
    #  2-leer descriptores de R de dir_descriptores_r
    #  3-para cada descriptor q localizar el mas cercano en R
    #     usar cdist o usar indices de busqueda eficiente
    #  3-crear dir_resultados_knn
    #    os.makedirs(dir_resultados_knn, exist_ok=True)
    #  4-escribir los knn en dir_resultados_knn
    #    incluir tambien los tiempos que representa cada ventana de q y r
    # borrar la siguiente linea
    print("ERROR: no implementado!")


# inicio de la tarea
if len(sys.argv) < 4:
    print("Uso: {} [dir_descriptores_q] [dir_descriptores_r] [dir_resultados_knn]".format(sys.argv[0]))
    sys.exit(1)

dir_descriptores_q = sys.argv[1]
dir_descriptores_r = sys.argv[2]
dir_resultados_knn = sys.argv[3]

tarea2_busqueda(dir_descriptores_q, dir_descriptores_r, dir_resultados_knn)
