import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn
import requests
import numpy as np
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import joblib

# Configurar opciones de la página
st.set_page_config(
    page_title="Clasificación COLELAP",
    page_icon="🏥",  
    layout="wide", 
    #initial_sidebar_state="collapsed"
)

def user_input_text():
    return st.text_area("Ingrese la descripción:")

import requests

def solicitud_API(muestra: list):
    # URL de la API
    urlApi = 'https://apiml2gei2023.azurewebsites.net/predict'

    # Preparar datos para la solicitud
    data = {
        "text": muestra  # Asumiendo que la API espera el texto de entrada como "text"
    }

    # Enviar solicitud POST a la API
    response = requests.post(urlApi, json=data)

    # Procesar la respuesta
    if response.status_code == 200:
        # La solicitud fue exitosa
        try:
            # Convertir la respuesta JSON en un diccionario
            result = response.json()

            # Extraer la predicción del diccionario
            prediction = result["prediction"]

            # Devolver la predicción
            return prediction
        except Exception as e:  # Manejar errores en la respuesta JSON
            print(f"Error al procesar la respuesta de la API: {e}")
            return None

    else:
        # La solicitud falló
        print(f"Error al llamar a la API: Código de estado {response.status_code}")
        return None

# Método para cargar el modelo desde un archivo .sav
def load_model(model_path):
    # Cargar el modelo desde el archivo .sav
    model = joblib.load(model_path)
    return model

# Método para cargar el modelo y el tokenizador desde el archivo .sav
def load_model_and_tokenizer(model_path):
    # Cargar el modelo y el tokenizador desde el archivo .sav
    model_tokenizer_dict = joblib.load(model_path)
    loaded_model = model_tokenizer_dict['model']
    loaded_tokenizer = model_tokenizer_dict['tokenizer']

    loaded_model.config.use_multiprocessing_for_evaluation = False

    return loaded_model, loaded_tokenizer

# Método para realizar predicciones
def predict(text, model, tokenizer):
    # Encode the input text
    encoded_text = tokenizer(text, return_tensors="pt")
    # Make predictions using the model
    with torch.no_grad():
        outputs = model(**encoded_text)
        predicted_class = torch.argmax(outputs.logits).item()
    
    return predicted_class

def main():
    folder = 'data'
    archivo_data = 'COLELAP_ETIQUETA_SES.xlsx'
    #data = pd.read_csv(folder + '/' + archivo_data, sep=',')
    data = pd.read_excel(folder + '/' + archivo_data)
    # Diccionario para mapear clases a etiquetas
    clase_etiqueta = {0: "Baja", 1: "Media", 2: "Alta"}
    d = data.copy()
    d.drop('Razon', axis=1, inplace=True)
    d.drop('Documento', axis=1, inplace=True)
    d.drop('Nombres', axis=1, inplace=True)
    d.drop('CUPS', axis=1, inplace=True)
    d.drop('Des CUPS', axis=1, inplace=True)
    d.drop('Fecha', axis=1, inplace=True)
    d.drop('Cod Pre', axis=1, inplace=True)
    d.drop('Dx Pre', axis=1, inplace=True)
    d.drop('Cod Pos', axis=1, inplace=True)
    d.drop('Dx Pos', axis=1, inplace=True)
    d.drop('Hallazgos', axis=1, inplace=True)
    #d.drop('Tiempo Cirugia', axis=1, inplace=True)
    d.drop('Inicio', axis=1, inplace=True)
    d.drop('Fin', axis=1, inplace=True)
    d.drop('Tipo Cirugia', axis=1, inplace=True)
    d.drop('Cirugia Urgencia', axis=1, inplace=True)
    d.drop('Cirugia Programada', axis=1, inplace=True)
    d.drop('Categoría', axis=1, inplace=True)
    d.drop('Complejidad', axis=1, inplace=True)
    
    # Diccionario para estandarizar formato desde excel
    mapeo = {'Allta': 'Alta', 'b': 'Baja', 'Baja ': 'Baja', 'Media ': 'Media', 'baja ': 'Baja', 'alta': 'Alta',
         'media': 'Media', 'media ': 'Media', 'ALTA': 'Alta', 'baja':'Baja', 'alta ':'Alta'}
    # Reemplazando la estandarización en la bd
    d['Juicio Experto'] = d['Juicio Experto'].replace(mapeo)

    #d['Juicio Experto'] = d['Juicio Experto'].replace(clase_etiqueta)
    #caracteristicas = d.drop(['Juicio Experto'], axis=1)
    image = Image.open("Images/logo-ses-hospital-universitario-de-caldas.png")

    # Define el tamaño máximo en píxeles (ancho, alto)
    max_width = 400
    max_height = 400

    # Calcula la nueva escala en función del tamaño máximo
    if image.width > max_width:
        scale = max_width / image.width
    else:
        scale = max_height / image.height

    # Redimensiona la imagen manteniendo la relación de aspecto
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    image = image.resize((new_width, new_height))

    # Ajusta el tamaño de la imagen para que se ajuste mejor al diseño
    #image = image.resize((50, 50))
    st.sidebar.image(image)

    # Barra lateral a la izquierda
    st.write("#  Clasificador COLELAP 🩺 ")

    #st.sidebar.title("🎯 Sistema de Predicción")
    option = st.sidebar.selectbox("Seleccionar Opción", ["Home", "Descriptiva", "Predicciones"])
    
     # Mostrar contenido según la opción seleccionada en la barra lateral
    if option == "Home":
        st.markdown("## Contexto:")
        st.markdown("Disponemos de datos que permiten clasificar si los pacientes SES presentan niveles de complejidad Alta, Media o Baja en función de la descripción quirúrjica.")
        st.markdown("## Descripción de las variables:")
        st.markdown("""
            * **Descripción Quirúrjica:** Detalle del procedimiento en lenguaje natural escrito por médicos expertos.
            """)
        # Agrega un salto de línea antes de la imagen
        st.markdown("\n")
        st.markdown("\n")
        # Ingresa la URL de la imagen que deseas mostrar
        imagen_url = "Images/colelap.webp"
        # Muestra la imagen en la página
        # Center the image using a container with CSS class
        col1, col2 = st.columns(2)  # Create two columns
        with col1:
            st.image(imagen_url, width=500)  # Display the image in the right column

        # Apply CSS styling to center the image within the column
        st.markdown("""
        <style>
        .stImage {
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """, unsafe_allow_html=True)
    elif option == "Descriptiva":
        st.write("¡Top 5!")
        st.dataframe(d.head())
        st.markdown("<hr style='margin-top: 2px; margin-bottom: 15px;'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            conteo = d['Juicio Experto'].value_counts()
            st.write("¡Distribución general de los datos!")
            st.dataframe(conteo)
            st.write("")
            # Configurar el estilo de la gráfica
            sns.set(style="whitegrid")
            # Crear una figura y un eje usando Matplotlib
            fig2, ax = plt.subplots(figsize=(6, 6))
            # Graficar la gráfica de barras usando Seaborn
            sns.countplot(x='Sexo', data=d, palette="mako_r", ax=ax)
            # Etiquetas y título
            ax.set_xlabel("Sexo (0 = Femenino, 1 = Masculino)")
            ax.set_ylabel("Cantidad")
            ax.set_title("Distribución de género")
            # Mostrar la figura en Streamlit
            st.pyplot(fig2)
        with col2:    
            # Graficar la distribución de los datos como una gráfica de torta
            st.write("")
            plt.rcParams['font.size'] = 13
            inidices = conteo.index.tolist()
            fig, ax = plt.subplots()
            ax.pie(list(conteo.values),labels = inidices, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  
            st.pyplot(fig)
            st.write("")
    elif option == "Predicciones":
        st.subheader("Clasificación")
        st.markdown("<hr style='margin-top: 2px; margin-bottom: 15px;'>", unsafe_allow_html=True)
        st.write("Por favor ingrese la descripción quirurjica del Paciente: ")  
        descripcion = "No complicaciones" #user_input_text()
        if st.button("Predecir"):
            try:
                if descripcion:
                    # Ruta al archivo del modelo .sav
                    model_path = "Models/modelo_y_tokenizador.sav"
                    
                    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_path)
                    
                    # Cargar el modelo
                    #loaded_model = load_model(model_path)
                    
                    # Realiza la predicción con tu modelo preentrenado (solicitud_API)                    
                    prediction = "Baja"#predict(descripcion, loaded_model, loaded_tokenizer)
    
                    # Mapea la predicción a una descripción
                    prediction_descriptions = {
                        "Baja": '✅ Baja: El nivel de complejidad de este procedimiento es Bajo.',
                        "Media": '⚠️ Media: El nivel de complejidad de este procedimiento es Moderado.',
                        "Alta": '❌ Alta: El nivel de complejidad de este procedimiento es Alto.'
                    }
                    
                    # Muestra el resultado de la predicción
                    if prediction in prediction_descriptions:
                        st.success(prediction_descriptions[prediction])
                    else:
                        st.warning("La predicción no tiene una descripción asociada.")
                else:
                    st.warning("⚠️ Por favor ingrese una descripción para realizar la predicción.")
            except Exception as e:
                st.error("❌ Ocurrió un error al realizar la predicción: " + str(e))
                raise RuntimeError("Error al cargar el modelo: {}".format(e))


if __name__ == "__main__":
    main()


