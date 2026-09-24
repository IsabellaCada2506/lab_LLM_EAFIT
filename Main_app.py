import streamlit as st
import pandas as pd
import numpy as np

from groq import Groq
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px


# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="LLM Explorer - Groq",
    page_icon="🤖",
    layout="wide"
)


# ==========================================================
# TÍTULO
# ==========================================================

st.title("🤖 LLM Explorer con Groq")

st.write(
    "Aplicación para explorar modelos de lenguaje, "
    "tokens, Token IDs, Bag of Words, similitud, "
    "embeddings y generación de texto."
)


# ==========================================================
# API KEY
# ==========================================================

st.sidebar.header("🔑 Configuración")

api_key = st.sidebar.text_input(
    "Ingresa tu API Key de Groq",
    type="password",
    placeholder="gsk_..."
)


# Si no hay API Key, detenemos la aplicación
if not api_key:

    st.info(
        "Ingresa tu API Key de Groq en el menú de la izquierda "
        "para comenzar."
    )

    st.stop()


# Crear cliente de Groq
client = Groq(api_key=api_key)


# ==========================================================
# MODELOS
# ==========================================================

st.sidebar.subheader("🤖 Modelo LLM")

modelos = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b"
]

modelo = st.sidebar.selectbox(
    "Selecciona el modelo",
    modelos
)


# ==========================================================
# PARÁMETROS DEL MODELO
# ==========================================================

st.sidebar.subheader("⚙️ Parámetros")

temperature = st.sidebar.slider(
    "Temperatura",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.1
)

max_tokens = st.sidebar.slider(
    "Máximo de tokens",
    min_value=50,
    max_value=2000,
    value=500,
    step=50
)

top_p = st.sidebar.slider(
    "Top P",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.1
)


# ==========================================================
# PESTAÑAS
# ==========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💬 Generación",
        "🔤 Tokens",
        "📚 Bag of Words",
        "📊 Similitud",
        "🧠 Embeddings"
    ]
)


# ==========================================================
# TAB 1 - GENERACIÓN DE TEXTO
# ==========================================================

with tab1:

    st.header("💬 Generación de texto")

    st.write(
        "Escribe una pregunta o instrucción y el modelo "
        "generará una respuesta."
    )

    prompt = st.text_area(
        "Escribe tu prompt:",
        placeholder="Explica qué es una red neuronal."
    )

    if st.button("🚀 Generar texto"):

        if not prompt:

            st.warning(
                "Por favor, escribe un prompt."
            )

        else:

            try:

                response = client.chat.completions.create(
                    model=modelo,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )

                respuesta = response.choices[0].message.content

                st.subheader("Respuesta del modelo")

                st.write(respuesta)


                # ------------------------------------------
                # INFORMACIÓN DE TOKENS
                # ------------------------------------------

                st.subheader("📊 Uso de tokens")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Tokens de entrada",
                    response.usage.prompt_tokens
                )

                col2.metric(
                    "Tokens de salida",
                    response.usage.completion_tokens
                )

                col3.metric(
                    "Tokens totales",
                    response.usage.total_tokens
                )


            except Exception as e:

                st.error(
                    f"Ocurrió un error al consultar Groq: {e}"
                )


# ==========================================================
# TAB 2 - TOKENS
# ==========================================================

with tab2:

    st.header("🔤 Tokens y Token IDs")

    st.write(
        "Los modelos de lenguaje no procesan directamente "
        "las palabras. Primero dividen el texto en tokens."
    )

    texto_tokens = st.text_area(
        "Escribe un texto:",
        value="Hola, me gusta la inteligencia artificial."
    )


    if st.button("🔍 Analizar tokens"):

        try:

            # Tokenizador
            tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-multilingual-cased"
            )

            # Convertir texto en tokens
            tokens = tokenizer.tokenize(
                texto_tokens
            )

            # Obtener IDs
            token_ids = tokenizer.convert_tokens_to_ids(
                tokens
            )


            # Crear tabla
            datos = pd.DataFrame(
                {
                    "Token": tokens,
                    "Token ID": token_ids
                }
            )


            st.subheader("Tokens encontrados")

            st.dataframe(
                datos,
                use_container_width=True
            )


            st.metric(
                "Cantidad de tokens",
                len(tokens)
            )


        except Exception as e:

            st.error(
                f"Ocurrió un error al analizar los tokens: {e}"
            )


# ==========================================================
# TAB 3 - BAG OF WORDS
# ==========================================================

with tab3:

    st.header("📚 Bag of Words")

    st.write(
        "Bag of Words representa los textos utilizando "
        "la frecuencia de las palabras."
    )


    texto1 = st.text_area(
        "Texto 1:",
        value="La inteligencia artificial aprende de los datos."
    )


    texto2 = st.text_area(
        "Texto 2:",
        value="La inteligencia artificial procesa información."
    )


    if st.button("📊 Crear Bag of Words"):

        documentos = [
            texto1,
            texto2
        ]


        # Crear vectorizador
        vectorizer = CountVectorizer()


        # Crear matriz
        matriz = vectorizer.fit_transform(
            documentos
        )


        # Obtener palabras
        palabras = vectorizer.get_feature_names_out()


        # Crear DataFrame
        df_bow = pd.DataFrame(
            matriz.toarray(),
            columns=palabras,
            index=[
                "Texto 1",
                "Texto 2"
            ]
        )


        st.subheader("Matriz Bag of Words")

        st.dataframe(
            df_bow,
            use_container_width=True
        )


        st.subheader("Palabras encontradas")

        st.write(
            ", ".join(palabras)
        )


# ==========================================================
# TAB 4 - SIMILITUD
# ==========================================================

with tab4:

    st.header("📊 Métrica de similitud")

    st.write(
        "Se utiliza la similitud del coseno para comparar "
        "los dos textos."
    )


    texto_a = st.text_area(
        "Texto A:",
        value="Me gusta aprender inteligencia artificial."
    )


    texto_b = st.text_area(
        "Texto B:",
        value="Me interesa estudiar inteligencia artificial."
    )


    if st.button("📐 Calcular similitud"):

        documentos = [
            texto_a,
            texto_b
        ]


        # Convertir textos en vectores
        vectorizer = CountVectorizer()

        matriz = vectorizer.fit_transform(
            documentos
        )


        # Calcular similitud
        similitud = cosine_similarity(
            matriz[0],
            matriz[1]
        )


        valor = similitud[0][0]


        st.subheader("Resultado")

        st.metric(
            "Similitud del coseno",
            f"{valor:.4f}"
        )


        st.progress(
            float(valor)
        )


        if valor >= 0.8:

            st.success(
                "Los textos tienen una similitud alta."
            )

        elif valor >= 0.5:

            st.info(
                "Los textos tienen una similitud media."
            )

        else:

            st.warning(
                "Los textos tienen una similitud baja."
            )


# ==========================================================
# TAB 5 - EMBEDDINGS
# ==========================================================

with tab5:

    st.header("🧠 Embeddings")

    st.write(
        "Un embedding representa un texto mediante "
        "valores numéricos."
    )

    st.info(
        "En esta demostración utilizamos una representación "
        "vectorial basada en las palabras del texto."
    )


    texto_embedding = st.text_area(
        "Escribe un texto:",
        value="La inteligencia artificial es una tecnología."
    )


    if st.button("🧮 Generar representación"):

        try:

            vectorizer = CountVectorizer()


            matriz = vectorizer.fit_transform(
                [texto_embedding]
            )


            vector = matriz.toarray()[0]


            palabras = (
                vectorizer
                .get_feature_names_out()
            )


            df_embedding = pd.DataFrame(
                {
                    "Palabra": palabras,
                    "Valor": vector
                }
            )


            st.subheader(
                "Representación vectorial"
            )


            st.dataframe(
                df_embedding,
                use_container_width=True
            )


            # ------------------------------------------
            # GRÁFICA
            # ------------------------------------------

            fig = px.bar(
                df_embedding,
                x="Palabra",
                y="Valor",
                title="Representación vectorial del texto"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        except Exception as e:

            st.error(
                f"Ocurrió un error generando la representación: {e}"
            )


# ==========================================================
# INFORMACIÓN
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "LLM Explorer desarrollado con Streamlit, "
    "Groq y herramientas de procesamiento de lenguaje natural."
)
