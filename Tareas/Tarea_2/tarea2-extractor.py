# CC5213 - TAREA 2
# 28 de septiembre de 2023
# Alumno: Andrés Calderón Guardia

import sys, os.path, subprocess, librosa, numpy

def calcular_descriptores_mfcc(archivo_wav, samples_por_ventana, samples_salto, dimension):
    # leer audio
    samples, sr = librosa.load(archivo_wav, sr=None)
    print("audio samples={} samplerate={} segundos={:.1f}".format(len(samples), sr, len(samples) / sr))
    # calcular MFCC
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=dimension, n_fft=samples_por_ventana, hop_length=samples_salto)
    # convertir a descriptores por fila
    descriptores = mfcc.transpose()
    # usualmente es buena idea borrar la(s) primera(s) columna(s)
    descriptores = numpy.delete(descriptores, 0, 1)
    return descriptores

def convertir_a_wav(archivo_audio, sample_rate, dir_temporal):
    archivo_wav = "{}/{}.{}.wav".format(dir_temporal, os.path.basename(archivo_audio), sample_rate)
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
    os.makedirs(dir_descriptores, exist_ok=True)
    # Crear descriptores y timestamps
    descriptores = []
    nombres = []
    tiempos = []
    for nombre in os.listdir(dir_audios):
        # Descriptor
        ruta_archivo = "{}/{}".format(dir_audios, nombre)
        archivo_wav = convertir_a_wav(ruta_archivo, 22050, dir_descriptores)
        descriptor = calcular_descriptores_mfcc(archivo_wav, 2048, 2048, 32)
        descriptores.extend(descriptor)

        # Timestamps
        for i in range(0, 2048 * descriptor.shape[0], 2048):
            nombres.append(nombre)
            tiempos.append(i / 22050)

    # Guardar descriptores en un archivo
    tipo = dir_audios.split("/")[-2]
    archivo_descriptor = "{}/{}".format(dir_descriptores, f"descriptores_{tipo}.npy")
    numpy.save(archivo_descriptor, descriptores)
    # Guardar timestamps en un archivo
    timestamps_descriptor = "{}/{}".format(dir_descriptores, f"timestamps_{tipo}.data")
    with open(timestamps_descriptor, 'w') as f:
        for i in range(0, len(tiempos)):
            print("{}\t{:7.2f}".format(nombres[i], tiempos[i]), file=f)

# inicio de la tarea
if len(sys.argv) < 3:
    print("Uso: {} [dir_audios] [dir_descriptores]".format(sys.argv[0]))
    sys.exit(1)

dir_audios = sys.argv[1]
dir_descriptores = sys.argv[2]

tarea2_extractor(dir_audios, dir_descriptores)