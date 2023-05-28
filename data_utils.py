import pandas as pd
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


def load_data():
    # all_files = ['../data/'+f for f in listdir('../data/') if isfile(join('../data/', f))]
    # classes = [i for i in all_files if i.startswith('../data/CLASS')]
    # df_class = pd.concat([pd.read_csv(filename, sep=';') for filename in classes])
    #
    # df_class = df_class[df_class['DTD'] > -1][
    #     ['SDAT_S', 'DD', 'DTD', 'FLT_NUM', 'SORG', 'SDST', 'SEG_CLASS_CODE', 'PASS_BK']]
    # df_class['SDAT_S'] = pd.to_datetime(df_class['SDAT_S'], format='%d.%m.%Y')
    # df_class['DD'] = pd.to_datetime(df_class['DD'], format='%d.%m.%Y')
    df_class = pd.read_parquet('../data/CLEAR_CLASS_TO_USE.parquet')
    print(df_class.shape)
    return df_class


def select_flt(df, sorg, sdst, flt_num, dd, seg_class, start_dt, end_dt):
    tmp = df[(df['SORG'] == sorg) &
              (df['SDST'] == sdst) &
              (df['FLT_NUM'] == flt_num) &
              (df['DD'] == dd) &
              (df['SEG_CLASS_CODE'] == seg_class) &
              (df['SDAT_S'] >= start_dt) &
              (df['SDAT_S'] <= end_dt)
    ]

    # plt.figure(figsize=(15, 6))
    # plt.bar(tmp['SDAT_S'], tmp['PASS_BK'])
    # plt.grid()
    # plt.title(f"Динамика бронирований для рейса {flt_num}, класс {seg_class}\n с {start_dt} по {end_dt}")
    # plt.xticks(rotation=90);
    return tmp.head(100)