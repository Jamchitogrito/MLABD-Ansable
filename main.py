import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Прогноз недвижимости",
    layout="wide"
)
st.title("Прогноз стоимости недвижимости")

tab1, tab2, tab3 = st.tabs(["Прогноз", "Дашборд", "Справка"])

@st.cache_data
def load_data():
    path = Path(__file__).parent / "data_for_modeling.pkl"
    if not path.exists():
        st.error(f"Файл не найден: {path}")
        return None

    bundle = pd.read_pickle(path)
    X_test_scaled = bundle["X_test_scaled"].copy()
    y_test = bundle["y_test"]
    scaler = bundle.get("scaler")
    continuous_cols = bundle.get("continuous_cols", [])

    # Разворачиваем scaled обратно в сырые
    if scaler is not None and continuous_cols:
        cols = [c for c in continuous_cols if c in X_test_scaled.columns]
        if cols:
            X_test_scaled[cols] = scaler.inverse_transform(X_test_scaled[cols])

    # Цена обратно в TRY
    X_test_scaled["price"] = np.expm1(y_test.values)

    return X_test_scaled

df = load_data()

with tab1:
    st.subheader("Характеристики объекта")

    col1, col2, col3 = st.columns(3)

    with col1:
        size = st.number_input("Площадь, м2", min_value=10.0, max_value=500.0, value=100.0, step=5.0)
        room_total = st.number_input("Комнат всего", min_value=1, max_value=15, value=3, step=1)
        building_age = st.number_input("Возраст здания", min_value=0, max_value=50, value=5, step=1)

    with col2:
        floor_no = st.number_input("Этаж", min_value=-4, max_value=30, value=5, step=1)
        total_floor_count = st.number_input("Этажей в доме", min_value=1, max_value=30, value=10, step=1)
        listing_type = st.selectbox(
            "Тип объявления",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Продажа", 2: "Аренда", 3: "Другое"}[x]
        )

    with col3:
        sub_type = st.selectbox(
            "Тип недвижимости",
            options=["Daire", "Rezidans", "Villa", "Müstakil Ev", "Kooperatif", "Yazlık"]
        )
        heating_type = st.selectbox(
            "Отопление",
            options=["Kombi (Doğalgaz)", "Klima", "Merkezi Sistem", "Kalorifer (Doğalgaz)", "Yok"]
        )
        tom = st.number_input("Срок размещения", min_value=0, max_value=180, value=30, step=1)


    if st.button("Рассчитать стоимость", type="primary"):
        if floor_no > total_floor_count and floor_no > 0:
            st.error("Этаж не может быть выше этажности дома")
        else:
            payload = {
                "size": float(size),
                "room_total": int(room_total),
                "building_age": int(building_age),
                "floor_no": int(floor_no),
                "total_floor_count": int(total_floor_count),
                "listing_type": int(listing_type),
                "subtype": sub_type,
                "heating": heating_type,
                "tom": int(tom),
            }

            try:
                with st.spinner(""):
                    response = requests.post(API_URL, json=payload, timeout=10)
                    response.raise_for_status()
                    result = response.json()

                st.success(f" Прогнозируемая стоимость: {result['price']:,.0f} TRY")
                st.write(f"Диапазон: {result['range_low']:,.0f} - {result['range_high']:,.0f} TRY")
                st.write(f"Цена за м²: {result['price_per_m2']:,.0f} TRY")
                st.caption(f"Модель: {result['model_type']}")
            except Exception as e:
                st.error(f"Что-то пошло не так: {e}")
with tab2:
    st.subheader("Дашборд")

    if df is None:
        st.error("Файл data_for_modeling.pkl не найден")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Средняя цена", f"{df['price'].mean():,.0f} TRY")
        c2.metric("Медианная цена", f"{df['price'].median():,.0f} TRY")
        c3.metric("Средняя площадь", f"{np.expm1(df['size_log']).mean():.1f} м²")

        st.write("### Цена от площади")
        sample = pd.DataFrame({
            "Площадь": np.expm1(df['size_log']),
            "Цена": df['price'],
        }).dropna().sample(min(500, len(df)), random_state=42)
        st.scatter_chart(sample, x="Площадь", y="Цена")

        if 'room_total' in df.columns:
            st.write("### Средняя цена по комнатам")
            by_rooms = df.groupby(df['room_total'].round())['price'].mean()
            st.bar_chart(by_rooms)

with tab3:
    st.subheader("Справка")
    st.write("Приложение для прогнозирования стоимости недвижимости.")

    st.subheader("Как пользоваться")
    st.write("1 Откройте вкладку Прогноз")
    st.write("2 Заполни характеристики объекта")
    st.write("3 Нажмите рассчитать стоимость")

    st.subheader("Поля ввода")
    st.write("Площадь - общая площадь М2")
    st.write("Комнат всего - количество комнат")
    st.write("Возраст здания - лет с постройки")
    st.write("Этаж - на каком этаже")
    st.write("Этажей в доме - этажность")
    st.write("Тип объявления - продажа / аренда")
    st.write("Тип недвижимости - Daire, Villa и др.")
    st.write("Отопление - тип отопления")
    st.write("Срок размещения - дней на рынке")

    st.write("Тип: RandomForestRegressor")
    st.write("Данные: ~251 000 объектов")
    st.write("Признаков: 39")

    st.subheader("Версия")
    st.write("3000")