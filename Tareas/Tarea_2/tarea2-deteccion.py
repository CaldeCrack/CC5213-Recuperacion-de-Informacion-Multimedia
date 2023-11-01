# CC5213 - TAREA 2
# 28 de septiembre de 2023
# Alumno: [nombre]

import sys
import os.path


def tarea2_deteccion(dir_resultados_knn, file_resultados_txt):
    if not os.path.isdir(dir_resultados_knn):
        print("ERROR: no existe directorio {}".format(dir_resultados_knn))
        sys.exit(1)
    elif os.path.exists(file_resultados_txt):
        print("ERROR: ya existe archivo {}".format(file_resultados_txt))
        sys.exit(1)
    # Implementar la deteccion
    #  1-leer resultados de knn en dir_resultados_knn
    #  2-buscar secuencias similares entre audios
    #  4-escribir en file_resultados_txt las detecciones encontradas
    #    seguir el formato: print("{}\t{}\t{}\t{}\t{}".format(radio, desde, largo, cancion, confianza))
    # borrar la siguiente linea
    print("ERROR: no implementado!")


# inicio de la tarea
if len(sys.argv) < 3:
    print("Uso: {} [dir_resultados_knn] [file_resultados_txt]".format(sys.argv[0]))
    sys.exit(1)

dir_resultados_knn = sys.argv[1]
file_resultados_txt = sys.argv[2]

tarea2_deteccion(dir_resultados_knn, file_resultados_txt)
