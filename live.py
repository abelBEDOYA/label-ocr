import cv2
from asociation import LabelORCR
import matplotlib.pyplot as plt

fields = ['part n', 'cantidad', 'proveedor', 'descripcion', 
            'lote q', 'serie(s)', 'ref. pdl', 'op:', 'fecha']

labelocr = LabelORCR(fields)

cap = cv2.VideoCapture(0)
matrix = False
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
if matrix:
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
# Bucle principal para capturar y mostrar video
while True:
    # Captura frame por frame
    ret, frame = cap.read()
    
    # Verifica si la captura fue exitosa
    if not ret:
        print("Error: No se pudo recibir frame. Saliendo...")
        break
    labelocr.inferenciar_imagen(frame)
    img_det = labelocr.dibujar_inferencia()
    # Muestra el frame en una ventana llamada 'Webcam'
    cv2.imshow('Webcam', img_det)
    if matrix:
        fig, ax,heatmap = labelocr.plotear_matriz((fig, ax))
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Espera 1 milisegundo para ver si se presiona la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera el objeto de captura y cierra las ventanas
cap.release()
cv2.destroyAllWindows()

