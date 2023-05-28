def get_flt_num(df, sorg, sdst):
    flt_nums = df[(df['SORG'] == sorg) & (df['SDST'] == sdst)]['FLT_NUM'].to_list()
    flt_nums = list(set(flt_nums))
    return flt_nums


def get_class(df, sorg, sdst, flt_num, dd):
    class_values = df[(df['SORG'] == sorg) &
              (df['SDST'] == sdst) &
              (df['FLT_NUM'] == flt_num) &
              (df['DD'] == dd)]['SEG_CLASS_CODE'].to_list()
    class_values = list(set(class_values))
    return class_values
