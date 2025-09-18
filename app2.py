# app.py
# Streamlit app: PDF -> DataFrame (codigo, nombre, cal_mun) + descarga CSV/XLSX
# Requisitos: pip install streamlit pdfplumber pandas openpyxl

import io
import re
import pdfplumber
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Historial → DataFrame", page_icon="📄", layout="wide")
st.title("📄 PDF de historia académica → DataFrame (código, materia, cal_mun)")

uploaded = st.file_uploader("Sube el PDF", type=["pdf"])

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def detect_columns(words, header_map=("Código", "Nombre", "Cal.num")):
    """
    Encuentra las posiciones X aproximadas de las columnas a partir de los encabezados.
    Devuelve dict con nombre_columna -> x_center
    """
    headers_found = {}
    for w in words:
        txt = w["text"].strip().upper()
        for key in header_map:
            if key.upper() in txt and key not in headers_found:
                x_center = (w["x0"] + w["x1"]) / 2
                headers_found[key] = x_center
    # En algunos PDFs la columna aparece como "Nombre asignatura"
    if "Nombre" not in headers_found:
        for w in words:
            if "ASIGNATURA" in w["text"].upper():
                headers_found["Nombre"] = (w["x0"] + w["x1"]) / 2
                break
    # En algunos PDFs la calificación aparece como "Cal.num." o "Cal.num"
    if "Cal.num" not in headers_found and "Cal.num." in headers_found:
        headers_found["Cal.num"] = headers_found["Cal.num."]
    return headers_found

def group_lines(words, y_threshold=3.0):
    """
    Agrupa palabras por línea usando la coordenada Y (tolerancia).
    Devuelve lista de listas de words.
    """
    # Orden por top (y0) y luego x0
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    current = []
    last_top = None
    for w in words_sorted:
        if last_top is None or abs(w["top"] - last_top) <= y_threshold:
            current.append(w)
            last_top = w["top"] if last_top is None else (last_top + w["top"]) / 2
        else:
            if current:
                lines.append(current)
            current = [w]
            last_top = w["top"]
    if current:
        lines.append(current)
    return lines

def assign_to_columns(line_words, col_x_map, slack=60):
    """
    Asigna las palabras de una línea a columnas según x_center más cercano.
    slack controla tolerancia horizontal.
    """
    buckets = { "Código": [], "Nombre": [], "Cal.num": [] }
    targets = []
    for k in ["Código", "Nombre", "Cal.num"]:
        if k in col_x_map:
            targets.append((k, col_x_map[k]))
    if not targets:
        return buckets

    for w in line_words:
        cx = (w["x0"] + w["x1"]) / 2
        # elegir la columna más cercana en X
        nearest = sorted(targets, key=lambda t: abs(cx - t[1]))[0]
        if abs(cx - nearest[1]) <= slack:
            buckets[nearest[0]].append(w["text"])
        else:
            # si queda fuera de tolerancia, asumimos que es parte de 'Nombre' (columna ancha)
            buckets["Nombre"].append(w["text"])
    return {k: normalize_space(" ".join(v)) for k, v in buckets.items()}

def is_header_like(row):
    h_tokens = {"CODIGO","CÓDIGO","NOMBRE","ASIGNATURA","CAL.NUM","CAL.NUM."}
    text = " ".join([row.get("Código",""), row.get("Nombre",""), row.get("Cal.num","")]).upper()
    return any(tok in text for tok in h_tokens)

def parse_calificacion_to_float(cal_str: str):
    """
    Extrae el valor numérico de 'cal_mun' como float (formato 4,4 (cuatro, cuatro)).
    Devuelve float o None si no se puede.
    """
    if not cal_str:
        return None
    # Busca el primer número con coma decimal o entero
    m = re.search(r"\b(\d{1}(?:,\d)?)\b", cal_str)
    if not m:
        return None
    num = m.group(1).replace(",", ".")
    try:
        return float(num)
    except:
        return None

if uploaded:
    with pdfplumber.open(uploaded) as pdf:
        rows = []
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            if not words:
                continue
            col_map = detect_columns(words)
            if not col_map:
                # Si no detecta encabezados, intentamos una heurística:
                # usar percentiles de x para 3 columnas
                xs = sorted([(w["x0"]+w["x1"])/2 for w in words])
                if len(xs) >= 3:
                    q = pd.Series(xs).quantile([0.2, 0.5, 0.8]).tolist()
                    col_map = {"Código": q[0], "Nombre": q[1], "Cal.num": q[2]}
                else:
                    continue

            lines = group_lines(words)
            for line in lines:
                bucketed = assign_to_columns(line, col_map)
                if not any(bucketed.values()):
                    continue
                if is_header_like(bucketed):
                    continue
                # Filtra líneas sin información clave
                has_any = any(bucketed.get(k) for k in ["Código","Nombre","Cal.num"])
                if not has_any:
                    continue
                rows.append({
                    "codigo": bucketed.get("Código","").strip(),
                    "nombre": bucketed.get("Nombre","").strip(),
                    "cal_mun": bucketed.get("Cal.num","").strip(),
                })

        # Limpieza y consolidación de líneas (unir nombres que se partieron en varias filas)
        cleaned = []
        for r in rows:
            # Quitar basura evidente
            if re.fullmatch(r"^[0-9]{1}$", r["cal_mun"] or ""):  # líneas con '0'/'1' de 'Aprobó' a veces se cuelan
                r["cal_mun"] = ""
            # Si la fila no tiene código pero sí nombre, fusión con la última con código
            if (not r["codigo"]) and cleaned:
                # concatenar al nombre previo
                cleaned[-1]["nombre"] = normalize_space(cleaned[-1]["nombre"] + " " + (r["nombre"] or ""))
                # si hay calificación y faltaba antes, completar
                if r["cal_mun"] and not cleaned[-1]["cal_mun"]:
                    cleaned[-1]["cal_mun"] = r["cal_mun"]
            else:
                cleaned.append(r)

        df = pd.DataFrame(cleaned)

        # Filtros finales: códigos válidos (8 dígitos) y nombre no vacío
        df = df[df["codigo"].str.fullmatch(r"\d{8}", na=False)]
        df = df[df["nombre"].str.len() > 0]

        # Si la calificación viene vacía en algunas filas, intenta rehidratar desde texto vecino (opcional).
        # Aquí nos quedamos con lo que se haya podido parsear en columna.
        # Crear columna numérica (opcional para verificación)
        df["cal_mun_num"] = df["cal_mun"].apply(parse_calificacion_to_float)

        # Mantener solo las 3 columnas solicitadas (pero ofrecemos vista con la numérica para control)
        show_df = df[["codigo", "nombre", "cal_mun"]].copy()

    def limpiar(texto):
        texto = re.sub(r"^(OBLIGATORIA|COMPLEMENTARIA|ELECTIVA HM)\s+S\s+\d+(?:\s+\d+)?\s+", "", texto)
        return texto
    
    show_df["limpia"] = [limpiar(texto) for texto in  show_df.nombre]

    def limpiar_materia(texto: str) -> str:
            patron = r"(\d+)\s+\d+$"
            return re.sub(patron, r"\1", texto)
    
    show_df["Asignatura"] = [limpiar_materia(texto) for texto in show_df.limpia]

    def extraer_nota(texto: str) -> float | None:
        """
        Extrae la nota numérica de un texto tipo:
        '4,4 M 4,6 (cuatro,seis)' → 4.6
        '5 de V 5 (cinco,cero)' → 5.0
        Devuelve None si no encuentra nota.
        """
        # buscar números con coma decimal (ej. 4,6)
        match = re.search(r"(\d+,\d+)", texto)
        if match:
            return float(match.group(1).replace(",", "."))
        
        # buscar número entero (ej. 5)
        match = re.search(r"\b(\d+)\b", texto)
        if match:
            return float(match.group(1))
        return None
    
    show_df["nota"] = [extraer_nota(texto) for texto in show_df.cal_mun]

    show_df2 = show_df[["codigo", "Asignatura", "nota"]]

    st.success(f"Se extrajeron {len(show_df2)} filas.")
    st.dataframe(show_df2, use_container_width=True)

    # Descargas
    csv_bytes = show_df2.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv_bytes, file_name="materias.csv", mime="text/csv")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        show_df2.to_excel(writer, index=False, sheet_name="materias")
    st.download_button("⬇️ Descargar Excel", data=output.getvalue(), file_name="materias.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Sube un PDF para comenzar. Ej.: historia académica con columnas ‘Código’, ‘Nombre asignatura’ y ‘Cal.num.’")

seminario_tomado = st.selectbox(
    "¿Ha tomado algún seminario?",
    ["Sí", "No"]
)

# Pregunta 2: ¿Cuál es el nombre del seminario?
# opciones fijas: ninguno, analítica, derecho
nombre_seminario = st.selectbox(
    "¿Cuál es el nombre del seminario?",
    ["Ninguno", "Analítica", "Derecho"]
)

# Mostrar valores seleccionados
st.write("Respuesta 1 (ha tomado?):", seminario_tomado)
st.write("Respuesta 2 (nombre del seminario):", nombre_seminario)



st.sidebar.title("Menú")
pagina = st.sidebar.radio("Navegación", ["Inicio", "Histograma"])

if pagina == "Inicio":
    st.title("Encuesta")
    hacer_hist = st.selectbox(
        "¿Quieres generar un histograma de la variable notas?",
        ["No", "Sí"]
    )
    if hacer_hist == "Sí":
        st.sidebar.success("Ve a la pestaña 'Histograma' para verlo 👈")

elif pagina == "Histograma":
    st.title("Histograma de Notas")
    fig, ax = plt.subplots()
    ax.hist(show_df2["nota"], bins=5, edgecolor="black")
    ax.set_xlabel("Notas")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)
