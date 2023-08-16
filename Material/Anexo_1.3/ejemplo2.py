# CC5213 - Recuperación de Información Multimedia
# Profesor: Juan Manuel Barrios
# Fecha: 08 de agosto de 2023

import sys
import os
import cv2

if len(sys.argv) <= 1:
    print("uso: {} [video.mp4 | id webcam 0,1,2]".format(sys.argv[0])) 
    sys.exit()

filename = sys.argv[1]
capture = None
if filename.isdigit():
    capture = cv2.VideoCapture(int(filename), cv2.CAP_DSHOW)
elif os.path.isfile(filename):
    capture = cv2.VideoCapture(filename)
if capture is None or not capture.isOpened():
    sys.exit()

cv2.namedWindow(filename + " (video)", cv2.WINDOW_AUTOSIZE)
while capture.grab():
    retval, frame_color = capture.retrieve()
    if not retval:
        continue
    frame_gris = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    thresh, frame_bin = cv2.threshold(frame_gris, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("frame_size={} binary_threshold={}".format(frame_bin.shape, thresh))
    cv2.imshow(filename + " (video)", frame_gris)
    cv2.imshow(filename + " (bin)", frame_bin)
    key = cv2.waitKey(33)
    if key == ord('q') or key == 27:
        break
capture.release()
cv2.destroyAllWindows()
