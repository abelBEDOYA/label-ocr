import difflib
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
from matplotlib import pyplot as plt # plot images
import cv2 #opencv
import os # folder directory navigation
from datetime import datetime, timedelta
import time
import numpy as np
import Levenshtein
import pandas as pd
import seaborn as sns


class LabelORCR:
    def __init__(self, fields: list[str], 
                 ignore_caps: bool = True, 
                 rel_levenshtein_thres: float= 0.4,
                 verbose: bool = True,
                 metric_criteria: str = 'debajo',
                 proximity_threshold = 0.02):
        """
        Initialize the FormParser with a list of field words.

        :param fields: List of field names or keywords to search for in OCR results.
        """
        self.fields = fields  # List of field words
        self.ignore_caps = ignore_caps
        self.rel_levenshtein_thres = rel_levenshtein_thres
        self.verbose = verbose
        self.metric_criteria = metric_criteria # debajo o deerecha o debajo_derecha
        self.proximity_threshold = proximity_threshold
        self.ocr_model = PaddleOCR(lang='en',cls=True)
        self.field_detections = {}  # Mapping from field to its detected OCR result
        self.unmatched_detections = []  # OCR detections not matched to any field
        self.field_values = {}  # Mapping from field to associated values
        self.last_df = None
        self.last_frame = None
        self.asociados={}
        self.asociados_default = {field: {'det_field': None, 'det_value': None} for field in self.fields}

    def inferenciar_imagen(self, frame):
        self.last_frame = frame
        self.last_detections = self.ocr_model.ocr(frame)[0]
        # print(self.last_detections)
        if self.last_detections is None:
            self.asociados = self.asociados_default
            self.last_df = None
            return 
        asociados_, det_field, det_value = self._asociar_cadenas(self.last_detections)
        self.last_df = self._recorrer_arrays_matriz_categorica(det_field, det_value)
        if self.verbose:
            print(self.last_df)

        self.asociados = self._obtener_parejas_valor_maximo(asociados_, det_field, det_value, self.last_df)
        return self.asociados
        
    def dibujar_detecciones(self, color = (0,0,255)):
        # Asegúrate de que la imagen esté en formato correcto (BGR)
        imagen_copy = self.last_frame.copy()
        if self.last_detections is None:
            return imagen_copy
        # Iteramos por cada detección
        for deteccion in self.last_detections:
            # Cada detección tiene la estructura: [coordenadas, (texto, confianza)]
            coords = deteccion[0]  # Coordenadas de la región
            texto, confianza = deteccion[1]  # Texto y confianza

            # Extraer las coordenadas del rectángulo
            x1, y1 = int(coords[0][0]), int(coords[0][1])
            x2, y2 = int(coords[2][0]), int(coords[2][1])

            # Dibujar el rectángulo en la imagen
            cv2.rectangle(imagen_copy, (x1, y1), (x2, y2), color, 2)

            # Colocar el texto detectado debajo del rectángulo
            # Definir la posición para el texto
            text_pos = (x1, y2 + 15)  # Debajo del rectángulo
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = color  # Color rojo para el texto
            thickness = 1
            
            # Colocar el texto en la imagen
            cv2.putText(imagen_copy, texto, text_pos, font, font_scale, font_color, thickness)

        return imagen_copy


    def _asociar_cadenas(self,detecciones):
        """Busca matcheos entre los campos del formulario/eiqeuta fotografiada (fields) 
        con los string detectados con el OCR (detecciones).
        Return:
            asociaciones: dict = estructura de la categorización en field o values, aunque soloc on los campos llenos.
            list = con las detecciones que son fields
            detecciones: list = con las demas detecciones"""
        
        asociaciones = {field: {'det_field': None, 'det_value': None} for field in self.fields}
        if self.verbose: print(f'Umbral de distancia: {self.rel_levenshtein_thres}')
        for field in self.fields:
            mejor_match = None
            mejor_distancia = float('inf')  # Empezamos con una distancia muy grande
            if len(detecciones)>0:
                for det in detecciones:
                    cadena_m = det[1][0]
                    if self.ignore_caps:
                        field_lower = field.lower()
                        cadena_m_lower = cadena_m.lower()
                    distancia = Levenshtein.distance(field_lower, cadena_m_lower)
                    longitud_max = max(len(field), len(cadena_m))
                    distancia_m = distancia/longitud_max
                    if distancia_m < mejor_distancia:
                        mejor_distancia = distancia_m
                        if distancia_m<self.rel_levenshtein_thres:
                            mejor_match = det
                if self.verbose:
                    print('Mejor parecido: \n \t Field:', field, '\n \t Deteccion: ', mejor_match[1][0] if mejor_match is not None else mejor_match, '\n \t Distancia: ', mejor_distancia)
                if mejor_match:
                    detecciones.remove(mejor_match)
            if mejor_match:
                asociaciones[field]['det_field'] = mejor_match
            else:
                asociaciones[field]['det_field'] = [[[-100,-100],[-120,-100],[-100,-120],[-120,-120]], (field, 0.0)]
        
        return asociaciones, [det['det_field'] for det in asociaciones.values() if det['det_field'] is not None], detecciones

    def _compute_M(self, field, value):
        """
        Compute the association metric M between two bounding boxes.

        :param bbox1: Bounding box of the first detection (x, y, w, h)
        :param bbox2: Bounding box of the second detection (x, y, w, h)
        :return: Computed metric M
        """
        if field is None or value is None:
            return -5
        # print('field: ',field[1][0], 'value: ', value[1][0])
        # Unpack bounding boxes
        esquinas = field[0]
        # Extraemos las coordenadas de las esquinas
        x11, y11 = esquinas[0]  # Esquina superior izquierda (aproximadamente)
        x12, y12 = esquinas[1]  # Esquina superior derecha
        x13, y13 = esquinas[2]  # Esquina inferior derecha
        x14, y14 = esquinas[3]  # Esquina inferior izquierda

        # Calcular el ancho y la altura del rectángulo
        # Suponiendo que las esquinas son de un rectángulo no rotado
        w1 = abs(x12 - x11)  # Ancho: diferencia entre las coordenadas x de las esquinas superior izquierda y superior derecha
        h1 = abs(y13 - y11)
        esquinas = value[0]
        # Extraemos las coordenadas de las esquinas
        x21, y21 = esquinas[0]  # Esquina superior izquierda (aproximadamente)
        x22, y22 = esquinas[1]  # Esquina superior derecha
        x23, y23 = esquinas[2]  # Esquina inferior derecha
        x24, y24 = esquinas[3]  # Esquina inferior izquierda

        # Calcular el ancho y la altura del rectángulo
        # Suponiendo que las esquinas son de un rectángulo no rotado
        w2 = abs(x22 - x21)  # Ancho: diferencia entre las coordenadas x de las esquinas superior izquierda y superior derecha
        h2 = abs(y23 - y21)

        # Compute Intersection over Union (IoU) for x-axis
        x_left = max(x11, x21)
        x_right = min(x11 + w1, x21 + w2)
        intersection_x = max(0, x_right - x_left)
        union_x = max(x11 + w1, x21 + w2) - min(x11, x21)
        IoU_x = intersection_x / union_x if union_x != 0 else 0
        # print('IoU_x', IoU_x)
        # Compute IoU for y-axis
        y_top = max(y11, y21)
        y_bottom = min(y11 + h1, y21 + h2)
        intersection_y = max(0, y_bottom - y_top)
        union_y = max(y11 + h1, y21 + h2) - min(y11, y21)
        IoU_y = intersection_y / union_y if union_y != 0 else 0
        # print('IoU_y', IoU_y)
        # print('-.----------------------')
        # Compute centers of the bounding boxes
        center_x1 = x11 + w1 / 2
        center_y1 = y11 + h1 / 2
        center_x2 = x21 + w2 / 2
        center_y2 = y21 + h2 / 2

        # Compute distances between centers
        dx = abs(center_x1 - center_x2)**1.3
        dy = abs(center_y1 - center_y2)**1.3

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-6
        por_encima = 1 if center_y2>y11 else -1
        por_izda = 1 if center_x2>x11 else -1
        # print('por_encima', por_encima)
        # print('-.----------------------')
        M = 0
        if self.metric_criteria == 'debajo' or self.metric_criteria=='debajo_derecha':
            M += por_encima*(max(h1,h2)*IoU_x / (dy + epsilon))
        if self.metric_criteria == 'derecha' or self.metric_criteria=='debajo_derecha':
            M += por_izda*por_encima*(max(w1,w2)*IoU_y / (dx + epsilon))
        if self.metric_criteria not in ['debajo', 'derecha', 'debajo_derecha']:
            M = por_izda*por_encima*(max(w1,w2)*IoU_y / (dx + epsilon)) + por_encima*(max(h1,h2)*IoU_x / (dy + epsilon))
        return M

    def _recorrer_arrays_matriz_categorica(self, det_field, det_value):
        # Crear una matriz vacía para almacenar los resultados
        matriz_resultados = np.zeros((len(det_field), len(det_value)))
        for i in range(len(det_field)):
            for j in range(len(det_value)):
                # Aplicar la métrica a cada par de elementos (array1[i], array2[j])
                matriz_resultados[i, j] = self._compute_M(det_field[i], det_value[j])
        # Crear un DataFrame de pandas, usando array1 como índices (filas) y array2 como columnas
        df_resultados = pd.DataFrame(matriz_resultados, index=[det[1][0] if det is not None else None for det in det_field], 
                                     columns=[det[1][0] for det in det_value])
        
        return df_resultados

    def _obtener_parejas_valor_maximo(self, asociados, det_fields, det_values, df_resultados):
        # Iteramos por cada fila en el DataFrame
        if df_resultados.empty:
            # print("Error: df_resultados está vacío")
            return self.asociados_default
        # print('det_values', det_values)
        for k, v in asociados.items():
            det_field = v['det_field'][1][0]
            # print('det_field', det_field)
            # print(df_resultados.index.tolist())
            # Encontramos el nombre de la columna con el valor máximo en esa fila
            column_value = df_resultados.loc[det_field].idxmax()
            valor_maximo = df_resultados.loc[det_field].max()
            try:
                if valor_maximo>self.proximity_threshold:
                    # Añadimos la pareja (index_value, column_value) a la lista de parejas
                    posibles_det_value = [det_value for det_value in det_values if det_value[1][0]==column_value]
                    if len(posibles_det_value)==1:
                        det_value = posibles_det_value[0]
                    else:
                        det_value = None
                else:
                    det_value = None
            except:
                det_value = None
            asociados[k]['det_value'] = det_value
        return asociados

    def dibujar_inferencia(self):
        imagen_copy = self.last_frame.copy()
        if self.asociados is None:
            return imagen_copy
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        # Iteramos por cada detección
        for k_field in self.asociados.keys():
            
            # Cada detección tiene la estructura: [coordenadas, (texto, confianza)]
            if self.asociados[k_field]['det_field'] is None:
                continue
            coords = self.asociados[k_field]['det_field'][0]  # Coordenadas de la región
            texto, confianza = self.asociados[k_field]['det_field'][1]  # Texto y confianza
            ## pintar field:
            # Extraer las coordenadas del rectángulo
            x11, y11 = int(coords[0][0]), int(coords[0][1])
            x12, y12 = int(coords[2][0]), int(coords[2][1])
            color = (255,0,0)
            # Dibujar el rectángulo en la imagen
            cv2.rectangle(imagen_copy, (x11, y11), (x12, y12), color, 2)

            text_pos = (x11, y12 + 15)  # Debajo del rectángulo
            font_color = color  # Color rojo para el texto
            # Colocar el texto en la imagen
            cv2.putText(imagen_copy, texto, text_pos, font, font_scale, font_color, thickness)

            # Cada detección tiene la estructura: [coordenadas, (texto, confianza)]
            if self.asociados[k_field]['det_value'] is None:
                continue
            coords = self.asociados[k_field]['det_value'][0]  # Coordenadas de la región
            texto, confianza = self.asociados[k_field]['det_value'][1]  # Texto y confianza
            ## pintar field:
            # Extraer las coordenadas del rectángulo
            x21, y21 = int(coords[0][0]), int(coords[0][1])
            x22, y22 = int(coords[2][0]), int(coords[2][1])
            color = (0,0,255)
            # Dibujar el rectángulo en la imagen
            cv2.rectangle(imagen_copy, (x21, y21), (x22, y22), color, 2)

            text_pos = (x21, y22 + 15)  # Debajo del rectángulo
            font_color = color  # Color rojo para el texto
            # Colocar el texto en la imagen
            cv2.putText(imagen_copy, texto, text_pos, font, font_scale, font_color, thickness)
            
            punto_inicial = (x11, y12)  # Coordenadas del punto inicial
            punto_final = (x21, y21)   # Coordenadas del punto final

            # Definir el color de la línea (BGR)
            color_linea = (0, 255, 0)  # Rojo en formato BGR
            ancho_linea = 5

            # Dibujar la línea
            cv2.line(imagen_copy, punto_inicial, punto_final, color_linea, ancho_linea)

        return imagen_copy
    
    def plotear_matriz(self, fig_ax=[None,None]):
        if fig_ax[0]:
            if self.last_df is None:
                return fig_ax[0], fig_ax[1], None
            elif self.last_df.empty:
                return fig_ax[0], fig_ax[1], None
            else:
                fig_ax[1].clear()
                heatmap = sns.heatmap(self.last_df, annot=True, cmap='coolwarm', fmt='.2f', vmin=-0.5, vmax=0.2, cbar=False, ax=fig_ax[1])
                return fig_ax[0], fig_ax[1], heatmap
        else:
            if self.last_df is None:
                return fig_ax[0], fig_ax[1], None
            elif self.last_df.empty:
                return fig_ax[0], fig_ax[1], None
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                heatmap = sns.heatmap(self.last_df, annot=True, cmap='coolwarm', fmt='.2f',vmin=-0.5, vmax=0.2, ax=ax)
                return fig, ax, heatmap

# Example usae:
if __name__ == "__main__":
    path = 'images/pladomin.jpg'
    img = cv2.imread(path)
    fields = ['part n', 'cantidad', 'proveedor', 'descripcion', 
              'lote q', 'serie(s)', 'ref. pdl', 'op:', 'fecha']

    labelocr = LabelORCR(fields)

    labelocr.inferenciar_imagen(img)
    labelocr.plotear_matriz()
    img_det = labelocr.dibujar_inferencia()
    cv2.imshow('Mi Imagen', img_det)

    # Esperar indefinidamente hasta que se presione una tecla
    cv2.waitKey(0)

    # parser.associate_ocr_results(result)

    # print("Field Detections:")
    # for field, detection in parser.field_detections.items():
    #     print(f"{field}: {detection}")

    # print("\nField Values:")
    # for field, values in parser.field_values.items():
    #     print(f"{field}: {values}")
