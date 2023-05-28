import streamlit as st
import pandas as pd
import numpy as np
from data_utils import select_flt
from filter_utils import get_flt_num, get_class
import datetime
from dateutil.relativedelta import relativedelta
from model_utils import get_model_data, get_model_data_future
import altair as alt
import zipfile

import warnings
warnings.filterwarnings('ignore')

st.title('Аэрофлот. Динамика бронирований')


@st.cache_data(persist=True)
def load_data():
    with zipfile.ZipFile('data/CLEAR_CLASS_TO_USE.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')
    st.write(1)
    try:
        data_class = pd.read_parquet('data/CLEAR_CLASS_TO_USE.parquet')
    except:
        st.write('oops)
    st.write(2)
    data_rasp = pd.read_csv('data/RASP2020.csv', sep=';').drop(
        columns=['NUM_LEGS', 'CAPTURE_DATE1', 'DEP_TIME1', 'ARR_TIME1', 'EQUIP1'])
    st.write(3)
    return data_class, data_rasp


df, data_rasp = load_data()
# st.write(df.head())
st.write(4)
app_mode = st.sidebar.selectbox('Раздел ',
                                ['Руководство', 'Динамика бронирований', 'Сезонность', 'Профили спроса', 'Прогноз', 'О команде'])

# выбираем команду
if app_mode == "Руководство":
    st.markdown('Как пользоваться')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,

        unsafe_allow_html=True,
    )


elif app_mode == 'Динамика бронирований':

    direction_mode = st.sidebar.selectbox('Направление',
                                          ['Москва - Сочи', 'Сочи - Москва', 'Москва - Астрахань', 'Астрахань - Москва'])

    try:
        # выбираем направление рейса
        if direction_mode == 'Москва - Сочи':
            sorg = 'SVO'
            sdst = 'AER'
        elif direction_mode == 'Сочи - Москва':
            sorg = 'AER'
            sdst = 'SVO'
        elif direction_mode == 'Москва - Астрахань':
            sorg = 'SVO'
            sdst = 'ASF'
        else:
            sorg = 'ASF'
            sdst = 'SVO'

        # выбираем рейс
        flt_nums = get_flt_num(df, sorg, sdst)
        flight = st.sidebar.selectbox('Рейс', flt_nums)

        # выбираем дату рейса
        flight_dd = st.sidebar.date_input("Дата рейса", datetime.date(2018, 1, 1), key='flt_date')

        # выбираем класс бронирования
        class_values = get_class(df, sorg, sdst, flight, flight_dd.strftime("%Y-%m-%d"))
        flight_class = st.sidebar.selectbox('Класс бронирования', class_values)

        # Выбираем интервал бронирований
        min_date = flight_dd + relativedelta(months=-1)
        max_date = flight_dd
        interval_date = st.sidebar.date_input("Период бронирования", (min_date, max_date), key = 'interval_dates')

        if st.sidebar.button('Вывести гистограмму'):
            df_to_plot_test = select_flt(df, sorg, sdst, flight, flight_dd.strftime("%Y-%m-%d"), flight_class,
                                         interval_date[0].strftime("%Y-%m-%d"), interval_date[1].strftime("%Y-%m-%d"))
            if df_to_plot_test.shape[0] == 0:
                st.exception('Нет рейсов, удовлетворяющих указанным параметрам. Измените фильтры')
            else:
                st.bar_chart(data=df_to_plot_test, x='SDAT_S', y='PASS_BK', width=0, height=0, use_container_width=True)
    except:
        st.success('Нет рейсов, удовлетворяющих указанным параметрам. Измените фильтры')

elif app_mode == 'Прогноз':
    direction_mode = st.sidebar.selectbox('Направление',
                                          ['Москва - Сочи', 'Сочи - Москва', 'Москва - Астрахань', 'Астрахань - Москва'])
    # выбираем направление рейса
    if direction_mode == 'Москва - Сочи':
        sorg = 'SVO'
        sdst = 'AER'
    elif direction_mode == 'Сочи - Москва':
        sorg = 'AER'
        sdst = 'SVO'
    elif direction_mode == 'Москва - Астрахань':
        sorg = 'SVO'
        sdst = 'ASF'
    else:
        sorg = 'ASF'
        sdst = 'SVO'

    # выбираем рейс
    flt_nums = get_flt_num(df, sorg, sdst)
    flight = st.sidebar.selectbox('Рейс', flt_nums)

    # выбираем класс бронирования
    class_values = ['B', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'X', 'Y', 'Z']
    flight_class = st.sidebar.selectbox('Класс бронирования', class_values)

    # Выбираем интервал прогноза
    min_date = datetime.datetime.now()
    max_date = min_date + relativedelta(months=1)
    interval_date = st.sidebar.date_input("Период прогноза", (min_date, max_date), key = 'interval_dates')

    if st.sidebar.button('Рассчитать прогноз'):
        st.write('...train model')
        if (interval_date[0].strftime("%Y-%m-%d") < '2020-01-01') and (interval_date[1].strftime("%Y-%m-%d") < '2020-01-01'):
            try:
                df_to_plot_test, df_to_plot_train, rmse_test, mae_test, min_train, max_train = get_model_data(df, sorg, sdst, flight, flight_class,
                                                 interval_date[0].strftime("%Y-%m-%d"), interval_date[1].strftime("%Y-%m-%d"))
                df_to_plot_test = df_to_plot_test.reset_index().melt('dd', var_name='Значение', value_name='y')
                df_to_plot_train = df_to_plot_train.reset_index().melt('dd', var_name='Значение', value_name='y')

                st.success(f'Ошибки прогноза: RMSE = {np.round(rmse_test, 2)}, MAE = {np.round(mae_test, 2)}')
                line_chart = alt.Chart(df_to_plot_test).mark_line().encode(
                    alt.X('dd', title='Точка прогнозирования количества бронирований'),
                    alt.Y('y', title='Количество бронирований'),
                    color='Значение:N'
                ).properties(
                    title=f"""Факт и предсказание бронирований с {interval_date[0].strftime("%Y-%m-%d")} по {interval_date[1].strftime("%Y-%m-%d")} (период прогноза)""",
                    width=1200
                )
                st.altair_chart(line_chart)

                line_chart_train = alt.Chart(df_to_plot_train).mark_line().encode(
                    alt.X('dd', title='Точка прогнозирования количества бронирований'),
                    alt.Y('y', title='Количество бронирований'),
                    color='Значение:N'
                ).properties(
                    title=f"""Факт и предсказание бронирований с {min_train.strftime("%Y-%m-%d")} по {max_train.strftime("%Y-%m-%d")} (период обучения модели)""",
                    width=1200
                )
                st.altair_chart(line_chart_train)
            except:
                st.write('Рейс не выполняет полеты в указанное время, выберите другое время или другой рейс')
        else:
            # try:
            df_to_plot_test, df_to_plot_train, min_train, max_train = get_model_data_future(df,
                                                                                            data_rasp, sorg, sdst,
                                                                                            flight,
                                                                                            flight_class,
                                                                                            interval_date[0].strftime("%Y-%m-%d"),
                                                                                            interval_date[1].strftime("%Y-%m-%d"))
            df_to_plot_test = df_to_plot_test.reset_index().melt('dd', var_name='Значение', value_name='y')
            df_to_plot_train = df_to_plot_train.reset_index().melt('dd', var_name='Значение', value_name='y')

            line_chart = alt.Chart(df_to_plot_test).mark_line().encode(
                alt.X('dd', title='Точка прогнозирования количества бронирований'),
                alt.Y('y', title='Количество бронирований'),
                color='Значение:N'
            ).properties(
                title=f"""Факт и предсказание бронирований с {interval_date[0].strftime("%Y-%m-%d")} по {interval_date[1].strftime("%Y-%m-%d")} (период прогноза)""",
                width=1200
            )
            st.altair_chart(line_chart)

            line_chart_train = alt.Chart(df_to_plot_train).mark_line().encode(
                alt.X('dd', title='Точка прогнозирования количества бронирований'),
                alt.Y('y', title='Количество бронирований'),
                color='Значение:N'
            ).properties(
                title=f"""Факт и предсказание бронирований с {min_train.strftime("%Y-%m-%d")} по {max_train.strftime("%Y-%m-%d")} (период обучения модели)""",
                width=1200
            )
            st.altair_chart(line_chart_train)
            # except:
            #     st.write('Рейс не выполняет полеты в указанное время, выберите другое время или другой рейс')

elif app_mode == "О команде":
    st.markdown('Градиентный подъем')
    st.markdown('')
    st.markdown('Это инфа про васьков, которые ничего не делают')
