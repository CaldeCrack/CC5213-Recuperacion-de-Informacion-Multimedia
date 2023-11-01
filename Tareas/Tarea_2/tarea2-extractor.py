# CC5213 - TAREA 2
# 28 de septiembre de 2023
# Alumno: [nombre]

import sys
import os.path
import subprocess


def convertir_a_wav(archivo_audio, sample_rate, dir_temporal):
    archivo_wav = "{}/{}.{}.wav".format(dir_temporal, os.path.basename(archivo_audio), sample_rate)
    # verificar si ya esta creado
    if os.path.isfile(archivo_wav):
        return archivo_wav
    comando = ["ffmpeg", "-i", archivo_audio, "-ac", "1", "-ar", str(sample_rate), archivo_wav]
    print("INICIANDO: {}".format(" ".join(comando)))
    proc = subprocess.run(comando, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise Exception("Error ({}) en comando: {}".format(proc.returncode, " ".join(comando)))
    return archivo_wav


def tarea2_extractor(dir_audios, dir_descriptores):
    if not os.path.isdir(dir_audios):
        print("ERROR: no existe directorio {}".format(dir_audios))
        sys.exit(1)
    elif os.path.exists(dir_descriptores):
        print("ERROR: ya existe directorio {}".format(dir_descriptores))
        sys.exit(1)
    # Implementar la tarea con los siguientes pasos:
    #  1-leer archivos de audio .m4a en dir_audios
    #  2-crear dir_descriptores
    #    os.makedirs(dir_descriptores, exist_ok=True)
    #  3-cada audio convertirlo a wav con ffmpeg
    #    por ejemplo: convertir_a_wav("archivo.m4a", 22050, dir_descriptores)
    #  4-cargar cada archivo wav
    #  5-calcular descriptores (ver material)
    #  6-escribir descriptores de cada audio en dir_descriptores
    # borrar la siguiente linea
    print("ERROR: no implementado!")


# inicio de la tarea
if len(sys.argv) < 3:
    print("Uso: {} [dir_audios] [dir_descriptores]".format(sys.argv[0]))
    sys.exit(1)

dir_audios = sys.argv[1]
dir_descriptores = sys.argv[2]

tarea2_extractor(dir_audios, dir_descriptores)
