def select_flt(df, sorg, sdst, flt_num, dd, seg_class, start_dt, end_dt):
    tmp = df[(df['SORG'] == sorg) &
              (df['SDST'] == sdst) &
              (df['FLT_NUM'] == flt_num) &
              (df['DD'] == dd) &
              (df['SEG_CLASS_CODE'] == seg_class) &
              (df['SDAT_S'] >= start_dt) &
              (df['SDAT_S'] <= end_dt)
    ]
    return tmp