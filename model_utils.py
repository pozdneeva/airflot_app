from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import timeit
import datetime
import streamlit as st

import warnings
warnings.filterwarnings('ignore')


def make_calendar(data):
    data['year_SDAT_S'] = data.index.year
    data['month_SDAT_S'] = data.index.month
    data['day_SDAT_S'] = data.index.day
    data['dayofweek_SDAT_S'] = data.index.dayofweek

    data['year_DD'] = data['DD'].dt.year
    data['month_DD'] = data['DD'].dt.month
    data['day_DD'] = data['DD'].dt.day
    data['dayofweek_DD'] = data['DD'].dt.dayofweek

    return data


def data_to_plot(y_test, predict):
    df_to_plot = pd.DataFrame()
    df_to_plot['PASS_BK'] = y_test
    df_to_plot['predict'] = np.round(predict)
    df_to_plot = df_to_plot.reset_index()
    df_to_plot.columns = ['dd', 'PASS_BK', 'predict']
    df_to_plot = df_to_plot.groupby('dd')[['PASS_BK', 'predict']].sum()
    return df_to_plot


def data_to_plot_score(df, predict):
    df_to_plot = pd.DataFrame()
    df_to_plot['DTD'] = df['DTD']
    df_to_plot['predict'] = np.round(predict)
    df_to_plot = df_to_plot.reset_index()
    df_to_plot = df_to_plot.drop(columns='DTD')
    df_to_plot.columns = ['dd', 'predict']
    df_to_plot = df_to_plot.groupby('dd')[['predict']].sum()
    return df_to_plot


def get_model_data(df, sorg, sdst, flt_num, seg_class, start_dt, end_dt):
    df = df[(df['SORG'] == sorg) &
            (df['SDST'] == sdst) &
            (df['FLT_NUM'] == flt_num) &
            (df['SEG_CLASS_CODE'] == seg_class) &
            (df['DTD'] >= 0)
        ]

    cols_to_use = ['DD', 'SDAT_S', 'DTD', 'PASS_BK']
    df = df[cols_to_use]
    df = df.set_index('SDAT_S')

    df = make_calendar(df)

    train_val_df = df[(df.index < start_dt)]
    train, valid = train_test_split(train_val_df, test_size=0.5, random_state=42)
    test = df[(df.index >= start_dt) & (df.index <= end_dt) ]

    X_train = train.drop(columns=['DD', 'PASS_BK'])
    X_valid = valid.drop(columns=['DD', 'PASS_BK'])
    X_test = test.drop(columns=['DD', 'PASS_BK'])

    y_train = train['PASS_BK']
    y_valid = valid['PASS_BK']
    y_test = test['PASS_BK']

    start_time = timeit.default_timer()
    cat = CatBoostRegressor(random_state=42, max_depth=6, learning_rate=0.06)
    cat.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)

    cat_predict_train = cat.predict(train_val_df.drop(columns=['DD', 'PASS_BK']))
    cat_predict = cat.predict(X_test)

    cat_res_train = mean_squared_error(train_val_df['PASS_BK'], cat_predict_train) ** 0.5
    cat_res = mean_squared_error(y_test, cat_predict) ** 0.5
    elapsed = timeit.default_timer() - start_time
    print('Execution Time for performance computation: %.2f minutes'%(elapsed/60))
    print('RMSE train:', cat_res_train)
    print('RMSE:', cat_res)

    df_to_plot_test = data_to_plot(y_test, cat_predict)
    df_to_plot_train = data_to_plot(train_val_df['PASS_BK'], cat_predict_train)

    mae_test = mean_absolute_error(y_test, cat_predict)
    min_train = train_val_df.index.min()
    max_train = train_val_df.index.max()

    return df_to_plot_test, df_to_plot_train, cat_res, mae_test, min_train, max_train


def separate_dates(row):
    res = []
    start_date = datetime.datetime.strptime(row['EFFV_DATE'], '%d.%m.%Y').date()
    end_date = datetime.datetime.strptime(row['DISC_DATE'], '%d.%m.%Y').date()

    # delta time
    delta = datetime.timedelta(days=1)

    # iterate over range of dates
    while (start_date <= end_date):
        res.append(start_date)
        start_date += delta

    return res


def generate_dates(row, start_date, end_date):
    res = []
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    # delta time
    delta = datetime.timedelta(days=1)

    # iterate over range of dates
    while (start_date <= end_date):
        res.append(start_date)
        start_date += delta

    return res


def have_flight(row):
    if str(row['weekday']) in str(row['FREQ']):
        return 1
    else:
        return 0


def prepare_rasp(df_test, sorg, sdst, flt_num, seg_class):
    df_test = df_test[(df_test['LEG_ORIG'] == sorg + ' ') &
                      (df_test['LEG_DEST'] == sdst + ' ') &
                      (df_test['FLT_NUMSH'] == flt_num)
                      ]
    df_test['SEG_CLASS_CODE'] = seg_class

    df_test['list_days'] = df_test.apply(lambda x: separate_dates(x), axis=1)
    df_test = df_test.explode('list_days')
    df_test['weekday'] = df_test['list_days'].apply(lambda x: x.weekday() + 1)
    df_test['is_in'] = df_test.apply(lambda x: have_flight(x), axis=1)
    df_test = df_test[df_test['is_in'] != 0].drop_duplicates().reset_index(drop=True)
    df_test = df_test[['FLT_NUMSH', 'SEG_CLASS_CODE', 'LEG_ORIG', 'LEG_DEST', 'list_days']].drop_duplicates()
    df_test.columns = ['FLT_NUM', 'SEG_CLASS_CODE', 'SORG', 'SDST', 'DD']
    return df_test


def generate_df_to_score(df, start_dt, end_dt):
    df['SDAT_S'] = df.apply(lambda x: generate_dates(x, start_dt, end_dt), axis=1)
    df = df.explode('SDAT_S')
    df['DD'] = pd.to_datetime(df['DD'], format='%d.%m.%Y')
    df['SDAT_S'] = pd.to_datetime(df['SDAT_S'], format='%d.%m.%Y')
    df['DTD'] = (df['DD'] - df['SDAT_S']).dt.days
    df = df[df['DTD'] >= 0]
    return df[['DD', 'SDAT_S', 'DTD']]


def get_model_data_future(df_train, df_test, sorg, sdst, flt_num, seg_class, start_dt, end_dt):
    # разбираемся с датасетом для обучения
    df_train = df_train[(df_train['SORG'] == sorg) &
            (df_train['SDST'] == sdst) &
            (df_train['FLT_NUM'] == flt_num) &
            (df_train['SEG_CLASS_CODE'] == seg_class) &
            (df_train['DTD'] >= 0)
        ]

    cols_to_use = ['DD', 'SDAT_S', 'DTD', 'PASS_BK']
    df_train = df_train[cols_to_use]
    df_train = df_train.set_index('SDAT_S')

    df_train = make_calendar(df_train)

    train_val_df = df_train[(df_train.index < start_dt)]
    train, valid = train_test_split(train_val_df, test_size=0.5, random_state=42)

    X_train = train.drop(columns=['DD', 'PASS_BK'])
    X_valid = valid.drop(columns=['DD', 'PASS_BK'])

    y_train = train['PASS_BK']
    y_valid = valid['PASS_BK']

    start_time = timeit.default_timer()
    cat = CatBoostRegressor(random_state=42, max_depth=6, learning_rate=0.06)
    cat.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
    cat_predict_train = cat.predict(train_val_df.drop(columns=['DD', 'PASS_BK']))
    cat_res_train = mean_squared_error(train_val_df['PASS_BK'], cat_predict_train) ** 0.5
    elapsed = timeit.default_timer() - start_time
    print('Execution Time for performance computation: %.2f minutes'%(elapsed/60))
    print('RMSE train:', cat_res_train)
    df_to_plot_train = data_to_plot(train_val_df['PASS_BK'], cat_predict_train)
    min_train = train_val_df.index.min()
    max_train = train_val_df.index.max()

    print('prepare test data')
    print(start_dt, end_dt)
    # get score data
    dff = prepare_rasp(df_test, sorg, sdst, flt_num, seg_class)
    dff = generate_df_to_score(dff, start_dt, end_dt)
    dff = dff.set_index('SDAT_S')
    dff = make_calendar(dff)
    # st.write(dff.head(10))

    cat_predict = cat.predict(dff)
    df_to_plot_test = data_to_plot_score(dff, cat_predict)
    return df_to_plot_test, df_to_plot_train, min_train, max_train
