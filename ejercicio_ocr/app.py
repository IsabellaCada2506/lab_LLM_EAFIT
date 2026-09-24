import streamlit as st
from openai import OpenAI
from PIL import Image
import pytesseract
import re
import math
from collections import Counter


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="OCR + GPT",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ OCR + LLM tipo GPT")
st.write(
    "Carga una imagen, extrae su texto mediante OCR y utiliza un LLM "
    "para ampliar y explicar la información."
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
    st.info("Ingresa tu API Key de OpenAI en el menú lateral para comenzar.")
    st.stop()

client = OpenAI(api_key=api_key)


# ============================================================
# CONFIGURACIÓN DEL MODELO
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

    with col1:
        st.subheader("Imagen cargada")
        st.image(
            image,
            caption="Imagen original",
            use_container_width=True
        )

    # ========================================================
    # OCR
    # ========================================================

    with col2:

        st.subheader("2. Texto extraído mediante OCR")

        if st.button("🔍 Extraer texto"):

            with st.spinner("Analizando imagen..."):

                try:

                    texto_ocr = pytesseract.image_to_string(
                        image,
                        lang="spa+eng"
                    )

                    texto_ocr = texto_ocr.strip()

                    if texto_ocr:

                        st.session_state["texto_ocr"] = texto_ocr

                    else:

                        st.warning(
                            "No se pudo encontrar texto en la imagen."
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

    st.text_area(
        "Texto detectado",
        texto_ocr,
        height=200
    )


    # ========================================================
    # GENERACIÓN DEL LLM
    # ========================================================

    st.header("3. Ampliar información con el LLM")

    if tipo_respuesta == "Formal":

        estilo = """
        Utiliza un lenguaje formal, claro y profesional.
        Explica la información de manera organizada.
        Evita utilizar expresiones demasiado informales.
        """

    else:

        estilo = """
        Utiliza un lenguaje técnico.
        Explica los conceptos importantes con precisión.
        Incluye detalles técnicos cuando sean relevantes.
        """

    prompt = f"""
    Analiza el siguiente texto obtenido mediante OCR:

    -------------------------
    {texto_ocr}
    -------------------------

    Amplía y explica la información contenida en el texto.

    {estilo}

    No inventes información que no esté relacionada con el texto.
    Organiza la respuesta utilizando párrafos y, cuando sea útil,
    listas o subtítulos.
    """

    if st.button("🤖 Generar respuesta con GPT"):

        with st.spinner("Generando respuesta..."):

            try:

                respuesta = client.chat.completions.create(
                    model=modelo,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un asistente especializado en "
                                "analizar y ampliar información obtenida "
                                "mediante OCR."
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

                texto_generado = respuesta.choices[0].message.content

                st.session_state["respuesta"] = texto_generado

                # Guardar métricas de tokens
                if respuesta.usage:

                    st.session_state["prompt_tokens"] = (
                        respuesta.usage.prompt_tokens
                    )

                    st.session_state["completion_tokens"] = (
                        respuesta.usage.completion_tokens
                    )

                    st.session_state["total_tokens"] = (
                        respuesta.usage.total_tokens
                    )

            except Exception as e:

                st.error(
                    f"Ocurrió un error al utilizar el LLM: {e}"
                )


# ============================================================
# RESPUESTA DEL LLM
# ============================================================

if "respuesta" in st.session_state:

    st.header("4. Respuesta ampliada")

    st.write(st.session_state["respuesta"])


    # ========================================================
    # MÉTRICAS
    # ========================================================

    st.header("5. Métricas del texto")

    texto_generado = st.session_state["respuesta"]

    # --------------------------------------------------------
    # Métricas básicas
    # --------------------------------------------------------

    palabras = re.findall(
        r"\b\w+\b",
        texto_generado,
        re.UNICODE
    )

    cantidad_palabras = len(palabras)

    cantidad_caracteres = len(texto_generado)

    cantidad_oraciones = len(
        re.findall(
            r"[.!?]+",
            texto_generado
        )
    )

    if cantidad_oraciones == 0:
        cantidad_oraciones = 1

    promedio_palabras_oracion = (
        cantidad_palabras / cantidad_oraciones
    )


    # --------------------------------------------------------
    # Diversidad léxica
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
    # Complejidad aproximada
    # --------------------------------------------------------

    longitud_promedio_palabra = (
        sum(
            len(palabra)
            for palabra in palabras
        )
        / cantidad_palabras
        if cantidad_palabras > 0
        else 0
    )


    # --------------------------------------------------------
    # Tokens del modelo
    # --------------------------------------------------------

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
    # MOSTRAR MÉTRICAS
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


    st.subheader("📊 Métricas lingüísticas")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Diversidad léxica",
            f"{diversidad_lexica:.2f}"
        )

    with col2:

        st.metric(
            "Promedio palabras/oración",
            f"{promedio_palabras_oracion:.2f}"
        )

    with col3:

        st.metric(
            "Longitud promedio palabra",
            f"{longitud_promedio_palabra:.2f}"
        )


    # ========================================================
    # MÉTRICAS SOLICITADAS POR EL EJERCICIO
    # ========================================================

    st.subheader("📝 Evaluación del texto")

    st.write(
        "Las siguientes métricas son indicadores aproximados "
        "calculados a partir de características del texto."
    )

    # --------------------------------------------------------
    # Coherencia
    # --------------------------------------------------------

    oraciones = re.split(
        r"[.!?]+",
        texto_generado
    )

    oraciones = [
        o.strip()
        for o in oraciones
        if o.strip()
    ]

    if len(oraciones) > 1:

        longitudes = [
            len(
                re.findall(
                    r"\b\w+\b",
                    oracion
                )
            )
            for oracion in oraciones
        ]

        promedio = sum(longitudes) / len(longitudes)

        diferencia = sum(
            abs(x - promedio)
            for x in longitudes
        ) / len(longitudes)

        coherencia = max(
            0,
            min(
                100,
                100 - diferencia * 2
            )
        )

    else:

        coherencia = 70


    # --------------------------------------------------------
    # Semántica
    # --------------------------------------------------------

    palabras_frecuentes = Counter(
        palabras_minusculas
    )

    palabras_repetidas = sum(
        1
        for palabra, cantidad
        in palabras_frecuentes.items()
        if cantidad > 1
    )

    if cantidad_palabras > 0:

        semantica = max(
            0,
            min(
                100,
                100 - (
                    palabras_repetidas
                    / cantidad_palabras
                    * 100
                )
            )
        )

    else:

        semantica = 0


    # --------------------------------------------------------
    # Sintaxis
    # --------------------------------------------------------

    oraciones_validas = 0

    for oracion in oraciones:

        palabras_oracion = re.findall(
            r"\b\w+\b",
            oracion
        )

        if len(palabras_oracion) >= 3:

            oraciones_validas += 1

    if len(oraciones) > 0:

        sintaxis = (
            oraciones_validas
            / len(oraciones)
            * 100
        )

    else:

        sintaxis = 0


    # --------------------------------------------------------
    # Gramática
    # --------------------------------------------------------

    errores_basicos = 0

    for oracion in oraciones:

        oracion = oracion.strip()

        if not oracion:
            continue

        if not oracion[0].isupper():

            errores_basicos += 1

    if len(oraciones) > 0:

        gramatica = max(
            0,
            100 - (
                errores_basicos
                / len(oraciones)
                * 100
            )
        )

    else:

        gramatica = 0


    # ========================================================
    # MOSTRAR INDICADORES
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Coherencia**")
        st.progress(
            int(coherencia)
        )
        st.write(
            f"{coherencia:.1f}/100"
        )

        st.write("**Semántica**")
        st.progress(
            int(semantica)
        )
        st.write(
            f"{semantica:.1f}/100"
        )

    with col2:

        st.write("**Sintaxis**")
        st.progress(
            int(sintaxis)
        )
        st.write(
            f"{sintaxis:.1f}/100"
        )

        st.write("**Gramática**")
        st.progress(
            int(gramatica)
        )
        st.write(
            f"{gramatica:.1f}/100"
        )


    # ========================================================
    # TOKENS
    # ========================================================

    st.subheader("🔢 Métricas de tokens")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Tokens del prompt",
            prompt_tokens
        )

    with col2:

        st.metric(
            "Tokens generados",
            completion_tokens
        )

    with col3:

        st.metric(
            "Tokens totales",
            total_tokens
        )
