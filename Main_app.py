```python
import streamlit as st
import pandas as pd
import re

from groq import Groq
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="LLM Explorer - Groq",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LLM Explorer con Groq")

st.write(
    "Explora tokenización, modelos LLM, temperatura, "
    "Bag of Words y similitud."
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

if not api_key:
    st.info("Ingresa tu API Key de Groq para comenzar.")
    st.stop()

client = Groq(api_key=api_key)


# ==========================================================
# MODELOS
# ==========================================================

st.sidebar.subheader("🤖 Modelo")

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
# PESTAÑAS
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💬 Generación",
        "🔤 Tokenización",
        "📚 Bag of Words",
        "📊 Similitud"
    ]
)


# ==========================================================
# TAB 1 - GENERACIÓN Y TEMPERATURA
# ==========================================================

with tab1:

    st.header("💬 Generación de texto")

    st.write(
        "Modifica la temperatura y observa cómo cambia "
        "la respuesta del modelo."
    )

    prompt = st.text_area(
        "Escribe tu prompt:",
        value="Explica qué es la inteligencia artificial."
    )

    st.subheader("🌡️ Temperatura")

    temperature = st.slider(
        "Selecciona la temperatura",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

    max_tokens = st.slider(
        "Máximo de tokens",
        min_value=50,
        max_value=1000,
        value=300,
        step=50
    )

    top_p = st.slider(
        "Top P",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.1
    )

    if st.button("🚀 Generar respuesta"):

        if not prompt:

            st.warning("Escribe un prompt.")

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

                st.subheader(
                    f"Respuesta con temperatura {temperature}"
                )

                st.write(respuesta)

                st.subheader("📊 Tokens utilizados")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Entrada",
                    response.usage.prompt_tokens
                )

                col2.metric(
                    "Salida",
                    response.usage.completion_tokens
                )

                col3.metric(
                    "Total",
                    response.usage.total_tokens
                )

            except Exception as e:

                st.error(f"Error: {e}")


# ==========================================================
# COMPARAR TEMPERATURAS
# ==========================================================

    st.divider()

    st.header("🔬 Comparar diferentes temperaturas")

    st.write(
        "Genera tres respuestas usando diferentes temperaturas "
        "para observar cómo cambia la generación."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        temp1 = st.number_input(
            "Temperatura 1",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )

    with col2:
        temp2 = st.number_input(
            "Temperatura 2",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1
        )

    with col3:
        temp3 = st.number_input(
            "Temperatura 3",
            min_value=0.0,
            max_value=2.0,
            value=1.5,
            step=0.1
        )

    if st.button("🔄 Comparar respuestas"):

        if not prompt:

            st.warning("Escribe un prompt primero.")

        else:

            temperaturas = [
                temp1,
                temp2,
                temp3
            ]

            for temp in temperaturas:

                try:

                    response = client.chat.completions.create(
                        model=modelo,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=temp,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )

                    respuesta = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    st.subheader(
                        f"🌡️ Temperatura: {temp}"
                    )

                    st.write(respuesta)

                    st.caption(
                        f"Tokens utilizados: "
                        f"{response.usage.total_tokens}"
                    )

                except Exception as e:

                    st.error(
                        f"Error con temperatura {temp}: {e}"
                    )


# ==========================================================
# TAB 2 - TOKENIZACIÓN
# ==========================================================

with tab2:

    st.header("🔤 Tokenización")

    st.write(
        "Aquí puedes observar cómo diferentes métodos "
        "dividen un texto en partes."
    )

    texto = st.text_area(
        "Escribe una frase:",
        value="Hola, me gusta aprender inteligencia artificial."
    )

    metodo = st.selectbox(
        "Método de tokenización",
        [
            "Por palabras",
            "Por caracteres",
            "Tokenizer de BERT"
        ]
    )


    # ------------------------------------------------------
    # FUNCIÓN PARA COLORES
    # ------------------------------------------------------

    def mostrar_tokens_colores(tokens):

        colores = [
            "#FFDDC1",
            "#C1FFD7",
            "#C1D4FF",
            "#F5C1FF",
            "#FFF3C1",
            "#D1C1FF",
            "#FFC1C1"
        ]

        html = ""

        for i, token in enumerate(tokens):

            color = colores[i % len(colores)]

            html += f"""
            <span style="
                background-color:{color};
                padding:6px;
                margin:3px;
                border-radius:6px;
                display:inline-block;
                border:1px solid #999;
            ">
                {token}
            </span>
            """

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    # ------------------------------------------------------
    # TOKENIZACIÓN
    # ------------------------------------------------------

    if st.button("🔍 Tokenizar texto"):

        if not texto:

            st.warning("Escribe un texto.")

        else:

            # ==============================================
            # POR PALABRAS
            # ==============================================

            if metodo == "Por palabras":

                tokens = re.findall(
                    r"\w+|[^\w\s]",
                    texto,
                    re.UNICODE
                )

                st.subheader("Tokens")

                mostrar_tokens_colores(tokens)

                st.write(
                    f"Cantidad de tokens: **{len(tokens)}**"
                )


            # ==============================================
            # POR CARACTERES
            # ==============================================

            elif metodo == "Por caracteres":

                tokens = list(texto)

                st.subheader("Caracteres")

                mostrar_tokens_colores(tokens)

                st.write(
                    f"Cantidad de caracteres: **{len(tokens)}**"
                )


            # ==============================================
            # TOKENIZER DE BERT
            # ==============================================

            else:

                try:

                    tokenizer = AutoTokenizer.from_pretrained(
                        "bert-base-multilingual-cased"
                    )

                    tokens = tokenizer.tokenize(texto)

                    token_ids = (
                        tokenizer.convert_tokens_to_ids(tokens)
                    )

                    st.subheader(
                        "Tokens del modelo"
                    )

                    mostrar_tokens_colores(tokens)

                    st.write(
                        f"Cantidad de tokens: **{len(tokens)}**"
                    )

                    st.subheader("Token IDs")

                    df = pd.DataFrame(
                        {
                            "Token": tokens,
                            "Token ID": token_ids
                        }
                    )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                except Exception as e:

                    st.error(
                        f"Error con el tokenizer: {e}"
                    )


# ==========================================================
# TAB 3 - BAG OF WORDS
# ==========================================================

with tab3:

    st.header("📚 Bag of Words")

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

        vectorizer = CountVectorizer()

        matriz = vectorizer.fit_transform(
            documentos
        )

        palabras = vectorizer.get_feature_names_out()

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


# ==========================================================
# TAB 4 - SIMILITUD
# ==========================================================

with tab4:

    st.header("📊 Similitud entre textos")

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

        vectorizer = CountVectorizer()

        matriz = vectorizer.fit_transform(
            documentos
        )

        similitud = cosine_similarity(
            matriz[0],
            matriz[1]
        )

        valor = similitud[0][0]

        st.metric(
            "Similitud del coseno",
            f"{valor:.4f}"
        )

        st.progress(
            float(valor)
        )
```
