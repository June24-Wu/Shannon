from feature.ops import FeatureOps

mysql_info = {
    'host': '172.16.1.13',
    'port': 3306,
    'user': 'dev_liguichuan',
    'password': '!*EeJZ5I4O8gdzPb8m-+',
    'db': 'test0415',
    'charset': 'utf8'
}
FeatureOps.generate_alpha_daily_ic_from_file(mysql_info, 'Investable', 'vwap')
