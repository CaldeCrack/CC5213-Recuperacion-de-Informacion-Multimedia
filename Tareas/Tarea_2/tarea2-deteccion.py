# CC5213 - TAREA 2
# 28 de septiembre de 2023
# Alumno: Andrés Calderón Guardia

import sys, os.path

def calcular_confianza(lista, valor, index, mean=False):
    suma = 0
    n = 0
    if mean: # media de los promedios
        for elem in lista:
            if elem[2] == valor:
                suma += float(elem[index])
                n += 1
    else: # promedio en K-NN
        for elem in lista:
            linea = elem.split("\t")
            if linea[2] == valor:
                suma += float(linea[index])
                n += 1
    return round(suma / n, 2)

def buscar_secuencias_similares(lista):
    canciones = []
    res = []
    K = 25
    densidad_minima = 15
    for linea in lista:
        canciones.append(linea.split("\t")[2])
    for i in range(0, len(canciones), K):
        densidades = {}
        upper_bound = K if len(canciones) - i > K else len(canciones) - i - 1
        for j in range(0, upper_bound):
            if canciones[i + j] in densidades:
                densidades[canciones[i + j]] += 1
            else:
                densidades[canciones[i + j]] = 1
        max_densidades = max(densidades, key=densidades.get)
        if densidades[max_densidades] > densidad_minima:
            for j in range(0, upper_bound):
                if canciones[i + j] == max_densidades:
                    p_indice = i + j
                    break
            for j in range(upper_bound, 0, -1):
                if canciones[i + j] == max_densidades:
                    u_indice = i + j
                    break
            confianza = calcular_confianza(lista[i:i + upper_bound], max_densidades, 4)
            cancion = [*lista[p_indice].split("\t")[:-1], p_indice, u_indice, confianza]
            res.append(cancion)
    return res

def condensar_duplicados_consecutivos(lista):
    ultimo = [None, None, None, None]
    j = 0
    res = []
    for i in range(0, len(lista)):
        linea = lista[i]
        if linea[2] != ultimo[2]:
            if ultimo[0] and i != j:
                res[-1][5] = ultimo[5]
                res[-1][6] = calcular_confianza(lista[j:i], ultimo[2], 6, mean=True)
            j = i
            res.append(linea)
        ultimo = linea
    return res

def tarea2_deteccion(dir_resultados_knn, file_resultados_txt):
    if not os.path.isdir(dir_resultados_knn):
        print("ERROR: no existe directorio {}".format(dir_resultados_knn))
        sys.exit(1)
    elif os.path.exists(file_resultados_txt):
        print("ERROR: ya existe archivo {}".format(file_resultados_txt))
        sys.exit(1)
    # leer archivo de knn
    archivo_knn = "{}/{}".format(dir_resultados_knn, "knn.data")
    with open(archivo_knn, 'r') as f:
        similares = f.readlines()

    # obtener las canciones encontradas y escribirlas
    secuencias = buscar_secuencias_similares(similares)
    canciones = condensar_duplicados_consecutivos(secuencias)
    with open(file_resultados_txt, 'w') as f:
        for i in range(0, len(canciones)):
            p_indice = int(canciones[i][4])
            u_indice = int(canciones[i][5])
            linea_inicial = similares[p_indice].split("\t")
            linea_final = similares[u_indice].split("\t")
            largo = round(float(linea_final[1])-float(linea_inicial[1]), 2)
            if largo <= 5.0 or 45 <= largo:
                continue
            radio = canciones[i][0]
            desde = canciones[i][1]
            cancion = linea_inicial[2]
            confianza = canciones[i][6]
            print("{}\t{}\t{}\t{}\t{}".format(radio, desde, largo, cancion, confianza), file=f)

# inicio de la tarea
if len(sys.argv) < 3:
    print("Uso: {} [dir_resultados_knn] [file_resultados_txt]".format(sys.argv[0]))
    sys.exit(1)

dir_resultados_knn = sys.argv[1]
file_resultados_txt = sys.argv[2]

tarea2_deteccion(dir_resultados_knn, file_resultados_txt)