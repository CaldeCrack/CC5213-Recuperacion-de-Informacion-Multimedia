# CC5213 - Recuperación de Información Multimedia
# Profesor: Juan Manuel Barrios
# Fecha: 08 de agosto de 2023

import sys
import cv2

if len(sys.argv) < 5:
    print("Uso:")
    print(" {}".format(sys.argv[0]))
    print("   id_webcam      Webcam. Ej: 0, 1, 2 ")
    print("   XML_detector   Archivo detector. Ej: haarcascade_frontalface_default.xml")
    print("   scale_factor   Reducción de la imagen. 1.01 (1%=lento), 1.1 (10%=normal), 1.5 (50%=rápido)")
    print("   min_neighbors  Cuantas veces se debe detectar. 1 (ruidoso), 5 (confiable), 20 (seguro)")
    sys.exit()

# parametros de entrada
id_webcam = int(sys.argv[1])
xml = sys.argv[2];
scale_factor = float(sys.argv[3])
min_neighbors = int(sys.argv[4])

# cargar el detector
file_xml = cv2.data.haarcascades + xml
print("cargando {}".format(file_xml))
detector = cv2.CascadeClassifier()
if not detector.load(file_xml):
    print("error leyendo {}".format(xml))
    sys.exit()
print("OK {}".format(xml))

# abrir webcam
print("abriendo webcam {}".format(id_webcam))
capture = cv2.VideoCapture(id_webcam, cv2.CAP_DSHOW)
if not capture.isOpened():
    print("error abriendo {}".format(id_webcam))
    sys.exit()
print("OK webcam {}".format(id_webcam))

# procesar frames
window_name = "webcam {}".format(id_webcam);
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
while capture.grab():
    retval, frame_color = capture.retrieve()
    if not retval:
        continue
    frame_gris = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    detecciones = detector.detectMultiScale(frame_gris, scale_factor, min_neighbors)
    print("detecciones: {}".format(len(detecciones)))
    for (x, y, w, h) in detecciones:
        center = (int(x + w / 2), int(y + h / 2))
        size = (int(w / 2), int(h / 2))
        color = (255, 255, 0)
        thickness = 3
        cv2.ellipse(frame_color, center, size, 0, 0, 360, color, thickness)
    cv2.imshow(window_name, frame_color)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
capture.release()
cv2.destroyAllWindows()
