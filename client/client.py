import streamlit as st
import tempfile
import numpy as np
from PIL import Image
from descriptor import glcm, bitdesc
import requests

# Configurar la página
st.set_page_config(layout="wide")

# Título de la aplicación
st.title("Récupération d'images basée sur le contenu")

# Dividir la pantalla en dos columnas
col1, col2 = st.columns(2)

# Variables para la lógica de búsqueda
url = 'http://127.0.0.1:8881/similarity'
descriptor = None
distancia = None
numero = None
image_chercher = None
similar = []
paths = [],
distances = []
labels = []

# Función para obtener la ruta del archivo temporal
def get_temporary_file_path(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    return temp_file.name

# Contenido de la columna izquierda
with col1:
    st.header("Sélection d'options de recherche")

    # Botones de radio para seleccionar el tipo de descriptor
    descriptor = st.radio(
        "Sélectionnez un type de descripteur:",
        ('glcm', 'bitdesc')
    )

    # Lista desplegable para seleccionar el tipo de distancia
    distancia = st.selectbox(
        "Sélectionnez un type de distance:",
        ('manhattan', 'euclidean', 'chebyshev', 'canberra')
    )

    # Caja de texto para escribir un número
    numero = st.number_input("Entrez le nombre d'images à afficher", min_value=1, value=5)
    
    st.header("Téléchargement d'images et affichage des résultats")

    # Cargar una imagen
    uploaded_file = st.file_uploader("Sélectionnez l'image à rechercher", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if uploaded_file is not None:
        # Obtener la ruta del archivo temporal
        image_path = get_temporary_file_path(uploaded_file)
        st.image(image_path, caption='Image pour la recherche', use_column_width=200)

    # Botón para iniciar la búsqueda
    if st.button('Chercher') and uploaded_file is not None:
        image_query = glcm(image_path) if descriptor == "glcm" else bitdesc(image_path)
        user_request = {
            'features': image_query,
            'descriptor': descriptor,
            'distances': distancia,
            'num_result': numero
        }
        response = requests.post(url, json=user_request)
with col2:
    try:
        if response.status_code == 200:
            st.header('Résultats de recherche')
            #print(f'Server returned: {response.json()}')
            response = response.json()
            for image_info in response['similar_image']:
                image = Image.open(image_info[0])
                st.image(image, caption=f'Image trouvée: {image_info[2]}', use_column_width=150)
        else:
            print(f'Failed to connect to the API. Status code: {response.status_code}')
    except Exception as e:
        print(f'Error: {e}')
    