# label-ocr
Codigo genérico para leer e interpretar información de formularios y etiquetas mediante vision artificial usando un OCR (Optical Character Recognition).


## USAGE:

Descargar:
```git clone https://github.com/abelBEDOYA/label-ocr.git```

Moverse a la carpeta descargada:
```cd label-ocr```

Instalar dependencias:
```pip install -r requirements.txt```


### Para usarlo en directo con una cámara conectada:
Ejecutar este script con la camara webcam en indice 0 enchufada:
```python live.py```

### Usarlo en un codigo:

```
## Importar la clase:
from asociation import LabelORCR

# Instanciar la clase con los campos a bsucar:
fields = ['part n (p)', 'cantidad', 'proveedor', 'descripcion', 
            'lote H', 'serie(s)', 'ref. pdl', 'op:', 'fecha']
labelocr = asociation.LabelORCR(fields)

# Abrir un imagen
import cv2
path = 'images/foto_etiqueta.jpg'
img = cv2.imread(path)

# Procesar la iamgen y obtener resutlados:
asociaciones = labelocr.inferenciar_imagen(img)
```

Los resultados tiene esta estructura:
```
print(asocciaciones)

{
 'proveedor': {'det_field': [[[1116.0, 300.0],
    [1316.0, 300.0],
    [1316.0, 334.0],
    [1116.0, 334.0]],
   ('proveedor (V):', 0.9428203701972961)],
  'det_value': [[[1110.0, 423.0],
    [1222.0, 423.0],
    [1222.0, 459.0],
    [1110.0, 459.0]],
   ('20168', 0.9983589053153992)]},
 'lote H': {'det_field': [[[1102.0, 474.0],
    [1220.0, 474.0],
    [1220.0, 508.0],
    [1102.0, 508.0]],
   ('Lote (H):', 0.9016936421394348)],
  'det_value': [[[1088.0, 584.0],
    [1196.0, 584.0],
    [1196.0, 617.0],
    [1088.0, 617.0]],
   ('28652', 0.9988812208175659)]},
...
 'fecha': {'det_field': [[[1043.0, 627.0],
    [1129.0, 632.0],
    [1127.0, 662.0],
    [1041.0, 657.0]],
   ('Fecha:', 0.9841441512107849)],
  'det_value': [[[1061.0, 669.0],
    [1187.0, 674.0],
    [1185.0, 717.0],
    [1059.0, 712.0]],
   ('26SEP', 0.9448102712631226)]}}
```
Eso es un diccionario donde cada key es un string de la lista de campos con los que se instanció la clase. El value es de neuvo un diccionario en cual tiene dos elementos con key "det_field" y "det_value", recogiendo en sus valores respectivamente la detección que se ha encontrado para el campo y el valor en el formulacion o etiqueta. Recordar que una deteccion tiene esta estructura, ejemplo:
````
 [[[1043.0, 627.0],[1129.0, 632.0], [1127.0, 662.0],[1041.0, 657.0]], # Coordenadas sobre la imagen del texto
   ('Fecha:', 0.9841441512107849)], # Caracteres detectados y confianza del modelo de 0 a 1.
````



## How does it work?

1. Se indican campos o caracteres que se deseean buscar: fecha, cantidad, nombre, apellido, ...
2. Se inferencia la imagen con un modelo de Deep Learning de OCR (Optical Character Recognition)
3. Se buscar de manera "flexible" mendiante la distancia relativa de Levensthein las asociaciones entre las deteccione y los campos indicados al princiio.
4. Las detecciones que no han sido tomadas en el paso anterior como campos son sometidas a una metrica de "proximidad" en relación a sus posiciones respecto a las de las detteciones tomadas como campos.
5. Con esta metrica se construye un matriz con dicha metrica entre todos los campos vs todas las demás detecciones.
6. Se toma como asociacion campo-valor los valores mas altos de la métrica de proximidad en la matriz que los enfrenta.
7. Estas asociaciones estan en el atributo accesible de la clase llamado: `asociados`