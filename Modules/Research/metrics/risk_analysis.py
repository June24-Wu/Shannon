# !/usr/bin/python3.7
# -*- coding: UTF-8 -*-
# @author: guichuan
import numpy as np


def calc_max_dd(net_value_list, date_list):
    max_unit_value = net_value_list[0]
    max_unit_value_flag = 0
    max_dd_flag_start = 0
    max_dd_flag_end = None
    max_dd = 0
    max_dd_daily = 0
    max_dd_daily_flag = None
    max_dd_date = None

    for i in range(1, len(net_value_list)):
        if net_value_list[i] >= max_unit_value:
            max_unit_value = net_value_list[i]
            max_unit_value_flag = i
        else:
            dd = net_value_list[i] / max_unit_value - 1
            dd_daily = net_value_list[i] / net_value_list[i-1] - 1
            if dd < max_dd:
                max_dd = dd
                max_dd_flag_start = max_unit_value_flag
                max_dd_flag_end = i
            if dd_daily < max_dd_daily:
                max_dd_daily = dd_daily
                max_dd_daily_flag = i

    if max_dd_daily_flag is not None:
        max_dd_date = date_list[max_dd_daily_flag]

    if max_dd_flag_end:
        return max_dd, (date_list[max_dd_flag_start], date_list[max_dd_flag_end]), max_dd_daily, max_dd_date
    else:
        return max_dd, (None, None), max_dd_daily, max_dd_date


def calc_sharpe_ratio(net_value_list, risk_free_return=0):
    ret_list = np.array(net_value_list[1:]) - np.array(net_value_list[0:-1])
    return (ret_list.mean() - risk_free_return) / ret_list.std() * np.sqrt(250)
