# CC5213 - Recuperación de Información Multimedia
## Profesor: Juan Manuel Barrios 
## Fecha: 28 de septiembre de 2023 

# Tarea 2

Para la tarea debe crear tres programas:

  * `tarea2-extractor.py`
     Recibe dos parámetros de la línea de comandos:
	    1. Una carpeta con audios. Procesa todos los archivos .m4a.
        2. Una carpeta donde guardar los wav temporales y los descriptores.
   
  * `tarea2-busqueda.py`
     Recibe tres parámetros de la línea de comandos:
	   1. La carpeta con descriptores de Q (radio).
	   2. La carpeta con descriptores de R (canciones).
	   3. La carpeta donde guardar el resultado de la comparación de Q y R.

  * `tarea2-deteccion.py`
     Recibe dos parámetros de la línea de comandos:
	   1. La carpeta del resultado de la comparación de Q y R.
	   2. El nombre del archivo a crear con el resultado de la detección de canciones.

El archivo de salida debe tener un formato de 5 columnas separadas por tabulador. En cada
fila debe tener el nombre de un audio de Q (radio), un tiempo de inicio y largo en segundos,
el nombre de un audio de R (canción) y un valor de confianza que la canción sea audible en
el segmento de radio indicado (mientras mayor, más seguro que sea correcto).

Por ejemplo, un posible archivo de resultados sería este:

radio-disney-ar-3.m4a	129.6	21.5	The Doors - Break on through (23).m4a	21
radio-disney-ar-3.m4a	813.8	12.6	Daft Punk - Around the world (13).m4a	32
radio-disney-ar-3.m4a	1559.2	40.4	Michael Jackson - Billie Jean (41).m4a	167
radio-disney-br-3.m4a	817.7	28.2	Celia Cruz - La Vida es un Carnaval (29).m4a	57
radio-disney-br-3.m4a	1363.6	32.5	Paul Anka - Diana (34).m4a	37
[......]


Para probar su tarea debe usar el programa de evaluación:

  `python evaluarTarea2.py`

Este programa llamará su tarea con los datasets ["a", "b", "c", "d"] y mostrará el resultado obtenido y la nota.

Para probar su tarea con un solo dataset 

  `python evaluarTarea1.py a`
  `python evaluarTarea1.py b`
  `python evaluarTarea1.py c`
  `python evaluarTarea1.py d`

Una detección es correcta si intersecta alguna detección del archivo gt.txt

Su tarea no puede demorar más de 30 minutos en evaluar un dataset.
