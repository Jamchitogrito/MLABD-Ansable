import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Прогноз недвижимости",
    layout="wide"
)
st.title("Прогноз стоимости недвижимости")

tab1, tab2, tab3 = st.tabs(["Прогноз", "Дашборд", "Справка"])

@st.cache_data
def load_data():
    path = Path("data/df1.pkl")
    if not path.exists():
        return None
    return pd.read_pickle(path)

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
            # Заглушка
            base_price = size * 15000
            age_factor = max(0.5, 1 - building_age * 0.02)
            floor_factor = 1 + 0.02 * max(0, floor_no)
            room_factor = 1 + 0.03 * room_total
            listing_factor = {1: 1.0, 2: 0.3, 3: 0.7}[listing_type]

            predicted = base_price * age_factor * floor_factor * room_factor * listing_factor

            st.success(f"### Прогнозируемая стоимость: **{predicted:,.0f} TRY**")
            st.write(f"Диапазон: {predicted * 0.85:,.0f} - {predicted * 1.15:,.0f} TRY")

            st.info("Это демонстрационный расчёт. Реальная модель будет подключена позже.")

with tab2:
    st.subheader("Статистика")

    if df is None:
        st.error("Данные не найдены.")
    else:
        st.success(f"Загружено {len(df):,} ")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Средняя цена", f"{df['price'].mean():,.0f} TRY")
        col2.metric("Медианная цена", f"{df['price'].median():,.0f} TRY")
        col3.metric("Средняя площадь", f"{df['size'].mean():.1f} М2")
        col4.metric("Средний возраст", f"{df['building_age'].mean():.1f} лет")

        st.subheader("Распределение цены")
        st.bar_chart(df['price'].head(100).reset_index(drop=True))

        st.markdown("---")

        st.subheader("Первые 20 строк")
        st.dataframe(df.head(20))

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

    st.write("Тип: Random Forest (заглушка)")
    st.write("Данные: ~251 000 объектов")
    st.write("Признаков: 39")

    st.subheader("Версия")
    st.write("3000")