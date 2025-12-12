import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Цена авто", layout="wide")

MODEL_PATH = "model.pkl"
TRAIN_CSV_PATH = "cars_train.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_train_ref():
    if os.path.exists(TRAIN_CSV_PATH):
        return pd.read_csv(TRAIN_CSV_PATH)
    return None


def to_num(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).split()[0])
    except:
        return np.nan


def clean_df(df):
    df = df.copy()
    df = df.drop(columns=["name", "torque"], errors="ignore")

    for col in ["mileage", "engine", "max_power"]:
        if col in df.columns:
            df[col] = df[col].apply(to_num)

    for col in ["year", "km_driven", "engine", "seats"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def train_medians():
    ref = load_train_ref()
    if ref is None:
        return None
    ref = clean_df(ref)
    return ref.select_dtypes(include=["int64", "float64"]).median()


def fill_missing(df, med):
    df = df.copy()
    for col in med.index:
        if col in df.columns:
            df[col] = df[col].fillna(med[col])
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")
    return df


model = load_model()
med = train_medians()
ref_train = load_train_ref()

t1, t2, t3 = st.tabs(["EDA", "Прогноз", "Веса модели"])

with t1:
    st.subheader("EDA")

    up = st.file_uploader("CSV для EDA", type="csv")
    df = pd.read_csv(up) if up is not None else ref_train

    if df is None:
        st.info("Загрузите CSV файл")
    else:
        df = fill_missing(clean_df(df), med)
        st.write("Размер данных:", df.shape)
        st.dataframe(df.head())

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        st.subheader("Распределение числового признака")
        num_sel = st.selectbox("Числовой признак", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[num_sel], bins=40, ax=ax)
        st.pyplot(fig)

        st.subheader("Распределение категориального признака")
        cat_sel = st.selectbox("Категориальный признак", cat_cols)
        fig, ax = plt.subplots()
        df[cat_sel].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.subheader("Pairplot числовых признаков")
        pair_cols = [c for c in num_cols if c != "selling_price"]
        sel_pair = st.multiselect("Выберите признаки", pair_cols, default=pair_cols[:3])
        if len(sel_pair) >= 2:
            sample = df[sel_pair].dropna()
            if len(sample) > 1000:
                sample = sample.sample(1000, random_state=42)
            g = sns.pairplot(sample)
            st.pyplot(g.fig)

        st.subheader("Корреляционная матрица")
        fig, ax = plt.subplots()
        sns.heatmap(df[num_cols].corr(), ax=ax)
        st.pyplot(fig)


with t2:
    st.subheader("Прогноз цены")

    up = st.file_uploader("CSV с признаками", type="csv", key="pred")
    if up is not None:
        df = pd.read_csv(up)
        df = fill_missing(clean_df(df), med)
        X = df.drop(columns=["selling_price"], errors="ignore")
        preds = model.predict(X)
        df["predicted_price"] = preds
        st.dataframe(df.head())
        st.download_button(
            "Скачать predictions.csv",
            df.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv",
        )

    st.markdown("---")
    st.subheader("Ручной ввод")

    ref = fill_missing(clean_df(ref_train), med)

    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Год выпуска", min_value=1980, max_value=2050)
        km_driven = st.number_input("Пробег", min_value=0)
        mileage = st.number_input("Расход топлива", min_value=0.0)
        engine = st.number_input("Объём двигателя", min_value=0.0)
        max_power = st.number_input("Мощность", min_value=0.0)

    with c2:
        seats = st.selectbox("Количество мест", sorted(ref["seats"].dropna().unique()))
        fuel = st.selectbox("Топливо", sorted(ref["fuel"].unique()))
        seller_type = st.selectbox("Тип продавца", sorted(ref["seller_type"].unique()))
        transmission = st.selectbox("Коробка передач", sorted(ref["transmission"].unique()))
        owner = st.selectbox("Владелец", sorted(ref["owner"].unique()))

    if st.button("Спрогнозировать"):
        one = pd.DataFrame([{
            "year": year,
            "km_driven": km_driven,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
        }])
        one = fill_missing(clean_df(one), med)
        price = model.predict(one)[0]
        st.success(f"Прогноз цены: {price:,.0f}".replace(",", " "))


with t3:
    st.subheader("Веса модели")

    if hasattr(model, "named_steps"):
        prep = model.named_steps["prep"]
        reg = model.named_steps["model"]

        names = prep.get_feature_names_out()
        coefs = reg.coef_.ravel()

        w = pd.DataFrame({"Признак": names, "Коэффициент": coefs})
        w["Коэф"] = w["Коэффициент"].abs()

        top_n = st.slider("Топ N признаков", 10, 50, 20)
        top = w.sort_values("Коэф", ascending=False).head(top_n)

        st.dataframe(top[["Признак", "Коэффициент"]])

        fig, ax = plt.subplots()
        ax.barh(top["Признак"], top["Коэффициент"])
        st.pyplot(fig)
