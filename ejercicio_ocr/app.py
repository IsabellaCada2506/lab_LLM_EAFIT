import streamlit as st
from openai import OpenAI
from PIL import Image
import easyocr
import numpy as np
import re


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="OCR + GPT",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ OCR + LLM tipo GPT")

st.write(
    "Carga una imagen, extrae su texto mediante OCR y utiliza "
    "un modelo GPT para ampliar y analizar la información."
)


# ============================================================
# API KEY
# ============================================================

st.sidebar.header("🔑 Configuración")

api_key = st.sidebar.text_input(
    "Ingresa tu API Key de OpenAI",
    type="password",
    placeholder="sk-..."
)

if not api_key:
    st.info(
        "Ingresa tu API Key de OpenAI en el menú lateral para comenzar."
    )
    st.stop()

client = OpenAI(api_key=api_key)


# ============================================================
# CONFIGURACIÓN DEL LLM
# ============================================================

st.sidebar.subheader("⚙️ Parámetros del LLM")

modelo = st.sidebar.selectbox(
    "Selecciona el modelo",
    [
        "gpt-4o-mini",
        "gpt-4o"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.1
)

max_tokens = st.sidebar.slider(
    "Máximo de tokens",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)

top_p = st.sidebar.slider(
    "Top P",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.1
)

tipo_respuesta = st.sidebar.radio(
    "Tipo de respuesta",
    [
        "Formal",
        "Técnica"
    ]
)


# ============================================================
# CARGAR MODELO OCR
# ============================================================

@st.cache_resource
def cargar_ocr():

    reader = easyocr.Reader(
        ["es", "en"],
        gpu=False
    )

    return reader


# ============================================================
# CARGAR IMAGEN
# ============================================================

st.header("1. Cargar imagen")

imagen = st.file_uploader(
    "Selecciona una imagen",
    type=["png", "jpg", "jpeg"]
)


if imagen is not None:

    image = Image.open(imagen)

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # IMAGEN
    # --------------------------------------------------------

    with col1:

        st.subheader("Imagen cargada")

        st.image(
            image,
            caption="Imagen original",
            use_container_width=True
        )


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    with col2:

        st.subheader("2. Extracción de texto")

        if st.button("🔍 Extraer texto"):

            with st.spinner(
                "Analizando imagen con OCR..."
            ):

                try:

                    # Cargar EasyOCR
                    reader = cargar_ocr()

                    # Convertir imagen a formato compatible
                    imagen_array = np.array(image)

                    # Ejecutar OCR
                    resultados = reader.readtext(
                        imagen_array
                    )

                    # Extraer solamente los textos
                    textos = []

                    for resultado in resultados:

                        texto_detectado = resultado[1]

                        textos.append(
                            texto_detectado
                        )

                    texto_ocr = " ".join(
                        textos
                    ).strip()

                    if texto_ocr:

                        st.session_state[
                            "texto_ocr"
                        ] = texto_ocr

                        st.success(
                            "Texto extraído correctamente."
                        )

                    else:

                        st.warning(
                            "No se encontró texto en la imagen."
                        )

                except Exception as e:

                    st.error(
                        f"Ocurrió un error durante el OCR: {e}"
                    )


# ============================================================
# MOSTRAR TEXTO OCR
# ============================================================

if "texto_ocr" in st.session_state:

    texto_ocr = st.session_state["texto_ocr"]

    st.subheader("Texto detectado por OCR")

    st.text_area(
        "Resultado del OCR",
        texto_ocr,
        height=200
    )


    # ========================================================
    # PREPARAR ESTILO DE RESPUESTA
    # ========================================================

    if tipo_respuesta == "Formal":

        estilo = """
        Utiliza un lenguaje formal, claro y profesional.
        Organiza bien la información.
        Utiliza explicaciones claras y evita expresiones informales.
        """

    else:

        estilo = """
        Utiliza un lenguaje técnico.
        Explica los conceptos con precisión.
        Incluye detalles técnicos cuando sean relevantes.
        Utiliza términos propios del área correspondiente.
        """


    # ========================================================
    # GENERAR RESPUESTA CON GPT
    # ========================================================

    st.header("3. Ampliar información con GPT")

    prompt = f"""
Analiza el siguiente texto obtenido mediante OCR:

-------------------------
{texto_ocr}
-------------------------

Amplía y explica la información contenida en el texto.

{estilo}

La respuesta debe:
- Explicar claramente la información.
- Mantener relación con el texto original.
- Organizar la información.
- No inventar información que no esté relacionada con el texto.
"""

    if st.button("🤖 Generar respuesta con GPT"):

        with st.spinner(
            "Generando respuesta..."
        ):

            try:

                respuesta = client.chat.completions.create(
                    model=modelo,

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un asistente especializado "
                                "en analizar textos obtenidos mediante OCR "
                                "y ampliar su contenido."
                            )
                        },

                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )


                texto_generado = (
                    respuesta
                    .choices[0]
                    .message
                    .content
                )


                # Guardar respuesta
                st.session_state[
                    "respuesta"
                ] = texto_generado


                # Guardar tokens
                if respuesta.usage:

                    st.session_state[
                        "prompt_tokens"
                    ] = respuesta.usage.prompt_tokens

                    st.session_state[
                        "completion_tokens"
                    ] = respuesta.usage.completion_tokens

                    st.session_state[
                        "total_tokens"
                    ] = respuesta.usage.total_tokens


            except Exception as e:

                st.error(
                    f"Ocurrió un error al utilizar GPT: {e}"
                )


# ============================================================
# MOSTRAR RESPUESTA
# ============================================================

if "respuesta" in st.session_state:

    texto_generado = st.session_state["respuesta"]

    st.header("4. Respuesta ampliada")

    st.write(
        texto_generado
    )


    # ========================================================
    # MÉTRICAS BÁSICAS
    # ========================================================

    st.header("5. Métricas del texto generado")


    # --------------------------------------------------------
    # Palabras
    # --------------------------------------------------------

    palabras = re.findall(
        r"\b\w+\b",
        texto_generado,
        re.UNICODE
    )

    cantidad_palabras = len(
        palabras
    )


    # --------------------------------------------------------
    # Caracteres
    # --------------------------------------------------------

    cantidad_caracteres = len(
        texto_generado
    )


    # --------------------------------------------------------
    # Oraciones
    # --------------------------------------------------------

    oraciones = re.split(
        r"[.!?]+",
        texto_generado
    )

    oraciones = [
        oracion.strip()
        for oracion in oraciones
        if oracion.strip()
    ]

    cantidad_oraciones = len(
        oraciones
    )


    # --------------------------------------------------------
    # Promedio de palabras por oración
    # --------------------------------------------------------

    if cantidad_oraciones > 0:

        promedio_palabras_oracion = (
            cantidad_palabras
            / cantidad_oraciones
        )

    else:

        promedio_palabras_oracion = 0


    # --------------------------------------------------------
    # Palabras únicas
    # --------------------------------------------------------

    palabras_minusculas = [
        palabra.lower()
        for palabra in palabras
    ]

    palabras_unicas = set(
        palabras_minusculas
    )


    if cantidad_palabras > 0:

        diversidad_lexica = (
            len(palabras_unicas)
            / cantidad_palabras
        )

    else:

        diversidad_lexica = 0


    # --------------------------------------------------------
    # Longitud promedio de palabra
    # --------------------------------------------------------

    if cantidad_palabras > 0:

        longitud_promedio = (
            sum(
                len(palabra)
                for palabra in palabras
            )
            / cantidad_palabras
        )

    else:

        longitud_promedio = 0


    # ========================================================
    # MÉTRICAS DE TOKENS
    # ========================================================

    prompt_tokens = st.session_state.get(
        "prompt_tokens",
        0
    )

    completion_tokens = st.session_state.get(
        "completion_tokens",
        0
    )

    total_tokens = st.session_state.get(
        "total_tokens",
        0
    )


    # ========================================================
    # MOSTRAR MÉTRICAS BÁSICAS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Palabras",
            cantidad_palabras
        )

    with col2:

        st.metric(
            "Caracteres",
            cantidad_caracteres
        )

    with col3:

        st.metric(
            "Oraciones",
            cantidad_oraciones
        )

    with col4:

        st.metric(
            "Tokens",
            total_tokens
        )


    st.subheader("📊 Otras medidas")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Diversidad léxica",
            f"{diversidad_lexica:.2f}"
        )

    with col2:

        st.metric(
            "Palabras por oración",
            f"{promedio_palabras_oracion:.2f}"
        )

    with col3:

        st.metric(
            "Longitud promedio",
            f"{longitud_promedio:.2f}"
        )


    # ========================================================
    # EVALUACIÓN CON EL MISMO LLM
    # ========================================================

    st.header(
        "6. Evaluación del texto generado"
    )

    st.write(
        "El modelo evalúa la respuesta generada "
        "en cuatro aspectos: coherencia, semántica, "
        "sintaxis y gramática."
    )


    if st.button(
        "📊 Evaluar texto"
    ):

        with st.spinner(
            "Evaluando el texto..."
        ):

            try:

                evaluacion_prompt = f"""
Evalúa el siguiente texto generado por un modelo de lenguaje:

-------------------------
{texto_generado}
-------------------------

Evalúa los siguientes aspectos de 0 a 100:

1. Coherencia:
Qué tan bien están conectadas y organizadas las ideas.

2. Semántica:
Qué tan claro es el significado y qué tan adecuadamente
se utilizan los conceptos.

3. Sintaxis:
Qué tan correctamente están construidas las oraciones.

4. Gramática:
Qué tan correctamente se utiliza la gramática.

Responde ÚNICAMENTE utilizando exactamente este formato:

Coherencia: número
Semántica: número
Sintaxis: número
Gramática: número

No agregues explicaciones.
"""


                evaluacion = client.chat.completions.create(
                    model=modelo,

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un evaluador de calidad "
                                "lingüística de textos."
                            )
                        },

                        {
                            "role": "user",
                            "content": evaluacion_prompt
                        }
                    ],

                    temperature=0
                )


                resultado_evaluacion = (
                    evaluacion
                    .choices[0]
                    .message
                    .content
                )


                # Guardar evaluación
                st.session_state[
                    "evaluacion"
                ] = resultado_evaluacion


            except Exception as e:

                st.error(
                    f"Ocurrió un error durante la evaluación: {e}"
                )


    # ========================================================
    # MOSTRAR EVALUACIÓN
    # ========================================================

    if "evaluacion" in st.session_state:

        evaluacion = (
            st.session_state["evaluacion"]
        )

        st.subheader(
            "Resultados de evaluación"
        )

        st.code(
            evaluacion
        )


        # ----------------------------------------------------
        # Extraer valores
        # ----------------------------------------------------

        coherencia = re.search(
            r"Coherencia:\s*(\d+)",
            evaluacion,
            re.IGNORECASE
        )

        semantica = re.search(
            r"Semántica:\s*(\d+)",
            evaluacion,
            re.IGNORECASE
        )

        sintaxis = re.search(
            r"Sintaxis:\s*(\d+)",
            evaluacion,
            re.IGNORECASE
        )

        gramatica = re.search(
            r"Gramática:\s*(\d+)",
            evaluacion,
            re.IGNORECASE
        )


        # ----------------------------------------------------
        # Convertir resultados
        # ----------------------------------------------------

        valor_coherencia = (
            int(coherencia.group(1))
            if coherencia
            else 0
        )

        valor_semantica = (
            int(semantica.group(1))
            if semantica
            else 0
        )

        valor_sintaxis = (
            int(sintaxis.group(1))
            if sintaxis
            else 0
        )

        valor_gramatica = (
            int(gramatica.group(1))
            if gramatica
            else 0
        )


        # ----------------------------------------------------
        # Mostrar barras
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Coherencia**")

            st.progress(
                min(valor_coherencia, 100)
            )

            st.write(
                f"{valor_coherencia}/100"
            )


            st.write("**Semántica**")

            st.progress(
                min(valor_semantica, 100)
            )

            st.write(
                f"{valor_semantica}/100"
            )


        with col2:

            st.write("**Sintaxis**")

            st.progress(
                min(valor_sintaxis, 100)
            )

            st.write(
                f"{valor_sintaxis}/100"
            )


            st.write("**Gramática**")

            st.progress(
                min(valor_gramatica, 100)
            )

            st.write(
                f"{valor_gramatica}/100"
            )


    # ========================================================
    # INFORMACIÓN DE LA CONFIGURACIÓN
    # ========================================================

    st.header(
        "7. Configuración utilizada"
    )

    configuracion = {
        "Modelo": modelo,
        "Temperature": temperature,
        "Max tokens": max_tokens,
        "Top P": top_p,
        "Tipo de respuesta": tipo_respuesta
    }

    st.table(
        configuracion
    )
