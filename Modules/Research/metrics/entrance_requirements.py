import numpy as np
import pandas as pd

from ..config.config import CORR_LIMIT, IC_LIMIT, IR_LIMIT, RET_LIMIT, EVAL_PERIODS, EVAL_PERIODS_IN_DAYS


def is_correlation_accepted(feature_name, corr_table):
    """
    judge correlation is accepted or not.
    """
    if corr_table is None:
        msg = f'{feature_name} already in feature db!'
        return 0, msg, None

    if corr_table is False:
        msg = f'Time out!{feature_name} test fails!'
        return 4, msg, None

    if corr_table.empty:
        msg = f'correlation test of {feature_name} passed. no feature in feature db!'
        return 1, msg, None

    temp = corr_table.loc[abs(corr_table['correlation'] >= CORR_LIMIT * 100)]
    if temp.empty:
        msg = f'correlation test of {feature_name} passed. correlation check num: {corr_table.shape[0]}'
        return 2, msg, None
    else:
        names = temp.index.tolist()
        msg = f'correlation test of {feature_name} fails, features: {names} are highly correlated!'
        return 3, msg, names


def is_ic_ir_accepted(feature_name, ic_table):
    """
    judge correlation is accepted or not.
    """
    if ic_table is None:
        msg = f'{feature_name} has non-reasonable ic table, failed to the test.'
        return False, msg, None
    ic_table.dropna(how='all', inplace=True)
    if ic_table.empty:
        msg = f'{feature_name} has empty ic table, failed to the test.'
        return False, msg, None

    msg = ""
    trading_direction = None

    # 3y test
    if len(ic_table) >= 753:
        ic_result_temp = ic_table.tail(753)
        ic_result_3y = ic_result_temp.mean().abs()
        ir_result_3y = (ic_result_temp.mean() / ic_result_temp.std()).abs()
        trading_direction = int(np.sign(ic_result_temp.mean().iloc[0]))
        for i in range(len(ic_result_3y)):
            if ic_result_3y.iloc[i] >= IC_LIMIT and ir_result_3y.iloc[i] >= IR_LIMIT:
                msg += f"3 year ICIR test of {feature_name} passed, " \
                       f"IC: {ic_result_3y.values.tolist()} | IR: {ir_result_3y.values.tolist()}"
            return True, msg, trading_direction
        else:
            msg += f"3 year ICIR test of {feature_name} failed, " \
                   f"IC: {ic_result_3y.values.tolist()} | IR: {ir_result_3y.values.tolist()}"

    # 2y test
    if len(ic_table) >= 502:
        ic_result_temp = ic_table.tail(502)
        ic_result_2y = ic_result_temp.mean().abs()
        ir_result_2y = (ic_result_temp.mean() / ic_result_temp.std()).abs()
        trading_direction = int(np.sign(ic_result_temp.mean().iloc[0]))
        for i in range(len(ic_result_2y)):
            if ic_result_2y.iloc[i] >= IC_LIMIT and ir_result_2y.iloc[i] >= IR_LIMIT:
                msg += f"2 year ICIR test of {feature_name} passed, " \
                       f"IC: {ic_result_2y.values.tolist()} | IR: {ir_result_2y.values.tolist()}"
            return True, msg, trading_direction
        else:
            msg += f"2 year ICIR test of {feature_name} failed, " \
                   f"IC: {ic_result_2y.values.tolist()} | IR: {ir_result_2y.values.tolist()}"

    # 1y test
    if len(ic_table) >= 252:
        ic_result_temp = ic_table.tail(251)
        ic_result_1y = ic_result_temp.mean().abs()
        ir_result_1y = (ic_result_temp.mean() / ic_result_temp.std()).abs()
        trading_direction = int(np.sign(ic_result_temp.mean().iloc[0]))
        for i in range(len(ic_result_1y)):
            if ic_result_1y.iloc[i] >= IC_LIMIT and ir_result_1y.iloc[i] >= IR_LIMIT:
                msg += f"1 year ICIR test of {feature_name} passed, " \
                       f"IC: {ic_result_1y.values.tolist()} | IR: {ir_result_1y.values.tolist()}"
            return True, msg, trading_direction
        else:
            msg += f"1 year ICIR test of {feature_name} failed, " \
                   f"IC: {ic_result_1y.values.tolist()} | IR: {ir_result_1y.values.tolist()}"

    return False, msg, trading_direction


def is_return_accepted(feature_name, return_table):
    return_flag = False
    return_list = []
    for length in EVAL_PERIODS_IN_DAYS:
        result_temp = return_table.tail(length)
        av = result_temp['alpha_value']
        _return = (av.iloc[-1] - av.iloc[0]) / av.iloc[0]
        _return_yearly = np.power(np.power(1 + _return, 1 / length), 252) - 1
        return_list.append(_return_yearly * 100)
    return_list_str = ','.join([str(i) for i in return_list])

    str_list = ','.join(EVAL_PERIODS)
    if min(return_list) >= RET_LIMIT:
        msg = f'return test of {feature_name} passed, return {str_list}: {return_list_str}'
        return_flag = True
    else:
        msg = f'return test of {feature_name} fails,  return {str_list}: {return_list_str}'

    return return_flag, msg, pd.Series(return_list, index=[f'ret_{item}' for item in EVAL_PERIODS], name='return_info')


def get_feature_status(feature_name, alphas_in_db, return_info, ic_table):
    _days = EVAL_PERIODS_IN_DAYS
    _periods = EVAL_PERIODS
    ic_factor_test = [ic_table.tail(i).mean().iloc[0] for i in _days]
    replace_reason = []

    count = 0
    watch_list_flag = False
    fail_to_beat_alphas = dict()
    fail_to_beat_ics = dict()
    fail_to_beat_rets = dict()
    succeed_to_beat_alphas = dict()
    succeed_to_beat_ics = dict()
    succeed_to_beat_rets = dict()

    for name in alphas_in_db.keys():
        ic_target = alphas_in_db[name]['ic']
        return_target = alphas_in_db[name]['ret']

        for i in range(len(_periods)):
            if abs(ic_factor_test[i]) > abs(ic_target[i]) and return_info[i] > return_target[i]:
                if i == 1:
                    count += 1

                    if _periods[i] in succeed_to_beat_alphas.keys():
                        succeed_to_beat_alphas[_periods[i]].append(name)
                        succeed_to_beat_ics[_periods[i]].append(str(ic_target[i]))
                        succeed_to_beat_rets[_periods[i]].append(str(return_target[i]))
                    else:
                        succeed_to_beat_alphas[_periods[i]] = []
                        succeed_to_beat_alphas[_periods[i]].append(name)
                        succeed_to_beat_ics[_periods[i]] = []
                        succeed_to_beat_ics[_periods[i]].append(str(ic_target[i]))
                        succeed_to_beat_rets[_periods[i]] = []
                        succeed_to_beat_rets[_periods[i]].append(str(return_target[i]))
                else:
                    continue

            else:
                if i == 1:
                    if _periods[i] in fail_to_beat_alphas.keys():
                        fail_to_beat_alphas[_periods[i]].append(name)
                        fail_to_beat_ics[_periods[i]].append(str(ic_target[i]))
                        fail_to_beat_rets[_periods[i]].append(str(return_target[i]))
                    else:
                        fail_to_beat_alphas[_periods[i]] = []
                        fail_to_beat_alphas[_periods[i]].append(name)
                        fail_to_beat_ics[_periods[i]] = []
                        fail_to_beat_ics[_periods[i]].append(str(ic_target[i]))
                        fail_to_beat_rets[_periods[i]] = []
                        fail_to_beat_rets[_periods[i]].append(str(return_target[i]))
                else:
                    continue

            if watch_list_flag is False:
                if abs(ic_factor_test[i]) > abs(ic_target[i]) or return_info[i] > return_target[i]:
                    watch_list_flag = True

    if count == len(alphas_in_db.keys()):
        succeed_to_beat_alphas_str = ','.join(succeed_to_beat_alphas[_periods[1]])
        succeed_to_beat_ics_str = ','.join(succeed_to_beat_ics[_periods[1]])
        succeed_to_beat_rets_str = ','.join(succeed_to_beat_rets[_periods[1]])
        change_reason = f"Succeeds to beat all highly correlated features:{succeed_to_beat_alphas_str}, " \
                        f"IC_1y: {ic_factor_test[1]}  vs {succeed_to_beat_ics_str}, " \
                        f"Return_1y: {return_info[1]} vs {succeed_to_beat_rets_str}"
        msg = f"{feature_name} will be accepted,  reason: {change_reason}"
        for i in range(len(succeed_to_beat_alphas[_periods[1]])):
            ic_temp = succeed_to_beat_ics[_periods[1]][i]
            ret_temp = succeed_to_beat_rets[_periods[1]][i]
            replace_reason_temp = f"Replace by {feature_name}, " \
                                  f"IC_1y: {ic_factor_test[1]}  vs {ic_temp}, " \
                                  f"Return_1y: {return_info[1]} vs {ret_temp}"
            replace_reason.append(replace_reason_temp)

        return 'accepted', change_reason, msg, replace_reason

    fail_to_beat_alphas_str = ','.join(fail_to_beat_alphas[_periods[1]])
    fail_to_beat_ics_str = ','.join(fail_to_beat_ics[_periods[1]])
    fail_to_beat_rets_str = ','.join(fail_to_beat_rets[_periods[1]])
    if watch_list_flag:
        change_reason = f"Fails to beat {fail_to_beat_alphas_str}, " \
                        f"IC_1y: {fail_to_beat_ics_str} vs {ic_factor_test[1]}, " \
                        f"Return_1y: {fail_to_beat_rets_str} vs {return_info[1]}"
        msg = f"{feature_name} will be watched,  reason: {change_reason}"
        return 'watched', change_reason, msg, replace_reason

    else:
        change_reason = f"Fails to beat all highly correlated features:{fail_to_beat_alphas_str}, " \
                        f"IC_1y: {fail_to_beat_ics_str} vs {ic_factor_test[1]}, " \
                        f"Return_1y: {fail_to_beat_rets_str} vs {return_info[1]}"
        msg = f"{feature_name} is not accepted,  reason: {change_reason}"
        return 'rejected', change_reason, msg, replace_reason
