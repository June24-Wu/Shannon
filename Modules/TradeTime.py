# !/usr/bin/python3.6
# -*- coding: UTF-8 -*-
import datetime
import DataAPI as api
import bisect
import calendar


class TradingTimeStock(object):
    def __init__(self, reference_time=None, freq: int = 3):
        self.freq = freq
        self.exchange = None
        self.reference_time = reference_time
        self.section_time = None
        self.time_now_loop = None
        if self.reference_time is None: self.reference_time = 93000000
        self.get_section_time()

    def get_section_time(self):
        """
        section: 交易区间, 对股票而言: 1代表上午连续交易时间 2 代表下午连续交易时间
                                 （！需要确认）对期货而言: 1代表开盘集合竞价 2代表上午第一段连续交易时间  3代表上午第二段连续交易时间  4 代表下午连续交易时间 5 收盘集合竞价
        :return: open_section, close_section 交易区间开始时间和结束时间
        """
        if self.reference_time is None:
            raise ValueError('Time(attribute: reference_time) has not been set, use method:set_time_benchmark() first!')

        result = dict()
        open_section = 93000000
        close_section = 113000000
        result['1'] = (open_section, close_section)
        open_section = 130003000
        close_section = 150000000
        result['2'] = (open_section, close_section)
        self.section_time = result

    def judge_section(self, query_time):
        open_section1, close_section1 = self.section_time['1']
        open_section2, close_section2 = self.section_time['2']
        if open_section1 <= query_time <= close_section1:
            section = 1
        elif open_section2 <= query_time <= close_section2:
            section = 2
        elif query_time < open_section1:
            section = 0
        elif close_section1 < query_time < open_section2:
            section = -1
        elif close_section2 < query_time:
            section = -2
        else:
            raise ValueError("无法识别此时间：%s" % str(query_time))
        return section

    @staticmethod
    def _convert(input_num):
        # print(input_num)
        if (input_num // 10 ** 3) % 100 >= 60:
            input_num -= 60000
            input_num += 100000
        if (input_num // 10 ** 5) % 100 >= 60:
            input_num -= 6000000
            input_num += 10000000
        return input_num

    @property
    def next_timestamp(self):
        if self.time_now_loop is None:
            self.time_now_loop = self.reference_time
        if self.time_now_loop is False:
            return None

        next_time = TradingTimeStock._convert(self.time_now_loop + self.freq * 1000)
        threshold = 2
        trading_section_list = list(range(1, threshold))
        no_trading_section_list = [0, -1]

        section_temp = self.judge_section(next_time)
        section = self.section
        if section_temp != section and section in trading_section_list:
            # section改变, 进入下一个section
            next_section = section + 1
            next_open, _ = self.section_time[str(next_section)]
            self.time_now_loop = next_open
            return next_open
        elif section_temp != section and section == threshold:
            # 交易结束
            self.time_now_loop = False
            return None
        elif section != section_temp and section in no_trading_section_list:
            # 非交易时段
            self.time_now_loop = False
            return None
        # elif self.section != section_temp and section == -5:
        #     # 非交易时段, 不可能出现
        elif section == section_temp and section in no_trading_section_list:
            # 非交易时段
            next_section = abs(section) + 1
            next_open, _ = self.section_time[str(next_section)]
            self.time_now_loop = next_open
            return next_open
        elif section == section_temp and section == -2:
            # 非交易时段
            self.time_now_loop = False
            return None
        else:
            self.time_now_loop = next_time
            return next_time

    @property
    def section(self):
        if self.time_now_loop is not None and self.time_now_loop is not False:
            return self.judge_section(self.time_now_loop)
        elif self.time_now_loop is False:
            return -2
        else:
            return None


class TradingTimeIndexFuture(object):
    def __init__(self, reference_time=None, freq: float = 0.5, flag_old_section: bool = False):
        self.freq = freq
        self.exchange = None
        self.reference_time = reference_time
        # for index future, trading time is changed at 2016-01-04
        self.flag_old_section = flag_old_section
        self.section_time = None
        self.time_now_loop = None
        if self.reference_time is None:
            if self.flag_old_section:
                self.reference_time = 91500000
            else:
                self.reference_time = 93000000
        self.get_section_time()

    def get_section_time(self):
        """
        section: 交易区间, 对股票而言: 1代表上午连续交易时间 2 代表下午连续交易时间
                                 （！需要确认）对期货而言: 1代表开盘集合竞价 2代表上午第一段连续交易时间  3代表上午第二段连续交易时间  4 代表下午连续交易时间 5 收盘集合竞价
        :return: open_section, close_section 交易区间开始时间和结束时间
        """
        if self.reference_time is None:
            raise ValueError('Time(attribute: reference_time) has not been set, use method:set_time_benchmark() first!')

        result = dict()
        if self.flag_old_section:
            open_section = 91500000
        else:
            open_section = 93000000
        close_section = 113000000
        result['1'] = (open_section, close_section)
        open_section = 130000000
        if self.flag_old_section:
            close_section = 151500000
        else:
            close_section = 150000000
        result['2'] = (open_section, close_section)
        self.section_time = result

    def judge_section(self, query_time):
        open_section1, close_section1 = self.section_time['1']
        open_section2, close_section2 = self.section_time['2']
        if open_section1 <= query_time <= close_section1:
            section = 1
        elif open_section2 <= query_time <= close_section2:
            section = 2
        elif query_time < open_section1:
            section = 0
        elif close_section1 < query_time < open_section2:
            section = -1
        elif close_section2 < query_time:
            section = -2
        else:
            raise ValueError("无法识别此时间：%s" % str(query_time))
        return section

    @staticmethod
    def _convert(input_num):
        if int(str(input_num)[-5:]) >= 60000:
            input_num -= 60000
            input_num += 100000
        if int(str(input_num)[-7:-5]) >= 60:
            input_num -= 6000000
            input_num += 10000000
        return input_num

    @property
    def next_timestamp(self):
        if self.time_now_loop is None:
            self.time_now_loop = self.reference_time
        if self.time_now_loop is False:
            return None

        next_time = TradingTimeIndexFuture._convert(self.time_now_loop + int(self.freq * 1000))
        threshold = 2
        trading_section_list = list(range(1, threshold))
        no_trading_section_list = [0, -1]

        section_temp = self.judge_section(next_time)
        section = self.section
        if section_temp != section and section in trading_section_list:
            # section改变, 进入下一个section
            next_section = section + 1
            next_open, _ = self.section_time[str(next_section)]
            self.time_now_loop = next_open
            return next_open
        elif section_temp != section and section == threshold:
            # 交易结束
            self.time_now_loop = False
            return None
        elif section != section_temp and section in no_trading_section_list:
            # 非交易时段
            self.time_now_loop = False
            return None
        # elif self.section != section_temp and section == -5:
        #     # 非交易时段, 不可能出现
        elif section == section_temp and section in no_trading_section_list:
            # 非交易时段
            next_section = abs(section) + 1
            next_open, _ = self.section_time[str(next_section)]
            self.time_now_loop = next_open
            return next_open
        elif section == section_temp and section == -2:
            # 非交易时段
            self.time_now_loop = False
            return None
        else:
            self.time_now_loop = next_time
            return next_time

    @property
    def section(self):
        if self.time_now_loop is not None and self.time_now_loop is not False:
            return self.judge_section(self.time_now_loop)
        elif self.time_now_loop is False:
            return -2
        else:
            return None


class ToDateUtils(object):

    @staticmethod
    def get_wtd_date(query_date):
        date_format = api.convert_to_date(query_date)
        weekday = date_format.weekday()
        monday_delta = datetime.timedelta(weekday)
        monday = date_format - monday_delta
        return monday

    @staticmethod
    def get_mtd_date(query_date):
        date_format = api.convert_to_date(query_date)
        return datetime.date(date_format.year, date_format.month, 1)

    @staticmethod
    def get_qtd_date(query_date):
        date_format = api.convert_to_date(query_date)
        qbegins = [datetime.date(date_format.year, month, 1) for month in (1, 4, 7, 10)]
        idx = bisect.bisect(qbegins, date_format)
        return qbegins[idx - 1]

    @staticmethod
    def get_ytd_date(query_date):
        date_format = api.convert_to_date(query_date)
        return datetime.date(date_format.year, 1, 1)


class ReportDateUtils(object):

    @staticmethod
    def get_report_date(query_date, count=1):
        date_format = api.convert_to_date(query_date)
        year = date_format.year
        month = date_format.month
        check = False

        while count > 0:
            if month % 3 == 0:
                if not check:
                    _, month_length = calendar.monthrange(year, month)
                    check = True
                    if date_format == datetime.date(year, month, month_length):
                        count -= 1
                    else:
                        if month - 3 <= 0:
                            month = month + 12 - 3
                            year -= 1
                        else:
                            month -= 3
                        count -= 1
                else:
                    if month - 3 <= 0:
                        month = month + 12 - 3
                        year -= 1
                    else:
                        month -= 3
                    count -= 1
            else:
                month_change = month % 3
                if month - month_change <= 0:
                    month = month + 12 - month_change
                    year -= 1
                else:
                    month -= month_change
                count -= 1
        _, month_length = calendar.monthrange(year, month)
        return datetime.date(year, month, month_length)

    @staticmethod
    def get_report_date_released(query_date, count=1):
        month_target = [5, 9, 11]
        date_format = api.convert_to_date(query_date)
        year = date_format.year
        month = date_format.month
        while count > 0:
            idx = bisect.bisect(month_target, month)
            if idx == 0:
                idx = 3
                year -= 1
            month = month_target[idx - 1]
            count -= 1
            if count != 0:
                month -= 1
        return datetime.date(year,  month, 1)


if __name__ == "__main__":
    trading_time = ReportDateUtils.get_report_date_released('2021-04-30', count=3)
    print(trading_time)
