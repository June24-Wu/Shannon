BASE_LIB_MYSQL = {
    'host': '172.16.1.13',
    'port': 3306,
    'user': 'shannon_read_only',
    'password': 'zBM+Yw8g(D#5ssZb',
    'db': 'base_lib',
    'charset': 'utf8'
}

FACTOR_LIB_MYSQL_TIO = {
    'host': '172.16.1.13',
    'port': 3306,
    'user': 'factor_lib_read_only',
    'password': 'Y^jZcJ+H8I^8Sf0f',
    'charset': 'utf8'
}

TEST_FACTORS_DAILY_TI0 = {
    'host': '172.16.4.1',
    'port': 3306,
    'user': 'fd_ti0_rw',
    'password': 'q&Zpiox8aigZ=k2n',
    'db': 'factors_daily_ti0',
    'charset': 'utf8'
}

# LIB PATH
LIB_PATH = r'/home/ShareFolder/lib_data'

# LIB PARAMETER
CORR_LIMIT = 0.75
IC_LIMIT = 2.0
IR_LIMIT = 0.2
COVER_RATE_LIMIT = 0.85
BT_STOCK_PCT = 0.1
BT_TRANSMISSION_RATE = 0.00025
BT_TAX_RATE = 0.0
RET_LIMIT = 5
EVAL_PERIODS = ('6m', '1y', '3y')
EVAL_PERIODS_IN_DAYS = (126, 251, 753)
