import os
import pandas as pd

# Función para cargar los datos
def loadData():
    # Obtiener la ruta del archivo xlsx
    file_path = os.path.dirname(__file__)
    # Construyendo la ruta completa al archivo
    full_path = os.path.join(file_path, '..', '..','Base_de_datos.xlsx')
    # Leer el archivo Excel
    df = pd.read_excel(full_path)
    return df