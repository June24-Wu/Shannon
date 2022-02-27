# !/usr/bin/python3.7
# -*- coding: UTF-8 -*-
# @author: guichuan
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm


def tanh(x, bound=1, alpha=0.5):
    """
    有两个参数要调节
    bound：上下限 一般取1 这样缩放到-1到1
    alpha: 调节变化速度 越大变化得越快 调试后发现取 一般取2/std, 这样到达2倍标准差的时候就比较接近1
    """
    y = bound * (1.0 - np.exp(-alpha * x)) / (1.0 + np.exp(-alpha * x))
    return y


def normfunc(x, mew=0, sigma=1):
    """
    正太分布的函数
    """
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-pow(x - mew, 2) / 2 / pow(sigma, 2))


def sigmoid(x, alpha=1, X=0):
    """
    sigmoid函数
    """
    y = 1 / (1 + np.exp(-alpha * (x - X)))
    return y


def thrfunc(x, thr_pos=0.5, alpha=80):
    """
    门函数
    大于thr_pos 的地方系数取值快速变为1
    小于thr_pos 的地方系数取值快速变为0
    一般把返回值当成X的系数
    """
    coef = sigmoid(abs(x) - thr_pos, alpha=alpha)
    return coef


def pow_enhance(sr, pow_num=0.5):
    """
    实验证明,如果要对序列求ema,那么做一个指数enhance后再做ema可以调高因子效果
    :param sr: 需要
    :param pow_num: 指数的系数, 小于1的数会衰减极端值, 大于1的数会增加
    :return:
    """
    return (abs(sr) ** pow_num) * np.sign(sr)


def sign_filter(sr):
    return np.sign(sr)


def clip_filter(sr, thr):
    """
    截尾操作防止极值,现在我觉得这个方法不如开根号增强
    :param sr:
    :param thr:
    :return:
    """
    return np.clip(sr, -thr, thr)


# EMA滤波
def ema_filter(sr, period=20):
    """
    EMA滤波方法
    :param sr:
    :param period:
    :return:
    """
    sr_ema = sr.ewm(span=period, adjust=False).mean()
    return sr_ema


def standard_scale(sr):
    if math.isclose(np.nanstd(sr), 0.0):
        return np.ones_like(sr)
    else:
        return (sr - np.nanmean(sr)) / np.nanstd(sr)


def rank(sr):
    return pd.Series(sr).rank(pct=True)


# 中位数滤波
def median_filter(sr, period=20):
    """
    中位数滤波法
    :param sr:
    :param period:
    :return:
    """
    sr_median = sr.rolling(period).median()
    sr_median = np.where(sr.median == np.nan, sr, sr_median)
    return sr_median


def calculate_medcouple_value(y):
    """
    Calculates the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.

    Returns
    -------
    mc : float
        The medcouple statistic

    Notes
    -----

    .. [*] M. Huberta and E. Vandervierenb, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """

    # Parameter changes the algorithm to the slower for large n

    y = np.squeeze(np.asarray(y))
    if y.ndim != 1:
        raise ValueError("y must be squeezable to a 1-d array")

    y = np.sort(y)

    n = y.shape[0]
    if n % 2 == 0:
        mf = (y[n // 2 - 1] + y[n // 2]) / 2
    else:
        mf = y[(n - 1) // 2]

    z = y - mf
    lower = z[z <= 0.0]
    upper = z[z >= 0.0]
    upper = upper[:, None]
    standardization = upper - lower
    is_zero = np.logical_and(lower == 0.0, upper == 0.0)
    standardization[is_zero] = np.inf
    spread = upper + lower
    h = spread / standardization
    # GH5395
    num_ties = int(np.sum(lower == 0.0))
    if num_ties:
        # Replacements has -1 above the anti-diagonal, 0 on the anti-diagonal,
        # and 1 below the anti-diagonal
        replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
        replacements -= 2 * np.triu(replacements)
        # Convert diagonal to anti-diagonal
        replacements = np.fliplr(replacements)
        # Always replace upper right block
        h[:num_ties, -num_ties:] = replacements

    return np.median(h)


def handle_extreme_value(y, method='MC'):
    """
    Handle extreme value.

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.
    method : MC -- MedCouple
           : MAD

    Returns
    -------
    y: 1-d array with extreme value handled
    low_change_rate: lower bound replace rate
    high_change_rate: higher bound replace rate
    """
    if len(y) == 0:
        raise ValueError("y cannot be empty!")
    if y.ndim != 1:
        raise ValueError("y must be 1-d array")

    if method == 'MC':
        mc = calculate_medcouple_value(y)
        value_25, value_75 = np.percentile(y, [25, 75])
        IQR = value_75 - value_25
        if mc >= 0:
            lowbound = value_25 - 1.5 * np.exp(-3.5 * mc) * IQR
            highbound = value_75 + 1.5 * np.exp(4 * mc) * IQR

        else:
            lowbound = value_25 - 1.5 * np.exp(-4 * mc) * IQR
            highbound = value_75 + 1.5 * np.exp(3.5 * mc) * IQR

        length = y.shape[0]
        low_change_rate = y[y < lowbound].shape[0] / length
        high_change_rate = y[y > highbound].shape[0] / length
        y[y < lowbound] = lowbound
        y[y > highbound] = highbound
        return y, low_change_rate, high_change_rate
    else:
        md = np.median(y)
        mad_ = np.median([np.abs(i - md) for i in y])
        if math.isclose(np.nanstd(mad_), 0.0):
            return y, None, None
        mad_e = 1.483 * mad_

        lowbound = md - 5 * mad_e
        highbound = md + 5 * mad_e
        low_change_rate = len(y[(y <= lowbound)]) / float(len(y))
        high_change_rate = len(y[(y >= highbound)]) / float(len(y))
        y[y < lowbound] = lowbound
        y[y > highbound] = highbound
        return y, low_change_rate, high_change_rate


def weighted_mean(s, weight):
    """weighted mean.
        s, weight are pd.Serie.
        sum of weight >0"""
    k = pd.concat([s, weight], axis=1)
    k = k.dropna()
    m = (k.iloc[:, 0] * k.iloc[:, 1]).sum() / k.iloc[:, 1].sum()
    return m


def weighted_var(s, weight):
    """weighted variance.
        s, weight are pd.Serie.
        sum of weight >0"""
    m = weighted_mean(s, weight)
    k = s - m
    k = k ** 2
    v = weighted_mean(k, weight)
    return v


def weighted_std(s, weight):
    """weighted std.
        s, weight are pd.Serie.
        sum of weight >0"""
    return np.sqrt(weighted_var(s, weight))


def rationalize_s(s, weight=None, sigma=3, fill=True, ifnan=True, fillvalue='mean'):
    """rationalize numeric serie, adjust abnormal value.
        sum of weight >0 if not None"""
    k = (s == np.inf) | (s == -np.inf)

    if type(weight) != type(None):
        m = weighted_mean(s[~k])
        sig = weighted_std(s[~k])
    else:
        m = s[~k].mean()
        sig = s[~k].std()

    if m == np.nan:
        raise ("{0} is empty".format(s.name))

    if fill:
        if fillvalue == 'mean':
            s[k | s.isna()] = m
        else:
            s[k | s.isna()] = 0.0
    else:
        s[k] = np.nan

    s[s > m + sigma * sig] = m + sigma * sig
    s[s < m - sigma * sig] = m - sigma * sig

    return s


def normalize_s(s, weight=None, sigma=3, fill=True, **kwargs):
    """normalize numeric serie, adjusted serie is norm(0,1).
        sum of weight >0 if not None"""
    ss = rationalize_s(s, weight, sigma, fill, **kwargs)

    if type(weight) != type(None):
        m = weighted_mean(s, weight)
        sig = weighted_std(s, weight)
    else:
        m = ss.mean()
        sig = ss.std()

    ss = (ss - m) / sig
    return ss


def neutralize(df1, df2, weight=None, normalize=True):
    """neutralize df1 with df2
        sum of weight >0 if not None"""
    n1 = len(df1.columns)
    n2 = len(df2.columns)
    if type(weight) != type(None):
        k = pd.concat([df1, df2, weight], axis=1)
        y = k.iloc[:, :n1]
        x = k.iloc[:, n1:n1 + n2]
        w = k.iloc[:, -1]
        m = sm.WLS(y, sm.add_constant(x), w)
        result = m.fit()
        t = result.resid
    else:
        k = pd.concat([df1, df2], axis=1)
        y = k.iloc[:, :n1]
        x = k.iloc[:, n1:n1 + n2]
        m = sm.OLS(y, sm.add_constant(x))
        result = m.fit()
        t = result.resid
    if normalize:
        t = t.apply(lambda x: normalize_s(x, weight))
    return t
