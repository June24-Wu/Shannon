from Platform_ops.utils.config_helper import Config
from Research.utils.namespace import load_namespace
import os

env = os.getenv("ENV")
env = 'dev'
config = Config()

# 钉钉通知相关
# prod
config.DING_SECRET_KEY = "SECbe7f9a39e866de9f9764e75ba78a2d7d05fed89d4f8d98f67d609693f0432b83"
config.DING_ACCESS_TOKEN = "9766e7bb07fb391a7eee1b2aa2cb47743555eed2156496ff9550f29ca94c3abe"

# # test
# config.DING_SECRET_KEY = "SEC2af0e6fd5df0953e44a9e84ebd8e4a6141dd5e14c6bc182a296db6779e3003cd"
# config.DING_ACCESS_TOKEN = "965543808b47781c15c7b65c8272fcc4819d88424eb0ce694b932fc19d0a5f77"

# 邮件相关
config.MAIL_HOST = 'smtp.exmail.qq.com'
config.MAIL_USER = 'shannon_ops@xinfei.cn'
config.MAIL_PWD = "gGhly9qrAb:d3.mb"
config.MAIL_SENDER = 'shannon_ops@xinfei.cn'
config.ENV_NAME = 'prod'

# SQL INFO
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

TOP_MYSQL_TIO = {
    'host': '172.16.1.13',
    'port': 3306,
    'user': 'dev_liguichuan',
    'password': '!*EeJZ5I4O8gdzPb8m-+',
    'db': 'factor_lib_ti0',
    'charset': 'utf8'
}

telephone_dict = {'liguichuan': '13916392907', 'sunyuting': '18813135183', 'fangxin': '13707665666',
                  'liangbin': '18601626312', 'xushengqiang': '13120929850', 'sunrui': '13552945080'}

config.ES = "172.16.2.11:9200"
config.ES_USER = "elastic"
config.ES_PASS = "shannon.1688"

# if env == "prod":
#     exec("import config.config_prod as prod")
#     exec("config.update(prod.__dict__)")
#     config.ENV_NAME = "prod"
# elif env == "test":
#     exec("import config.config_test as test")
#     exec("config.update(test.__dict__)")
#     config.ENV_NAME = "test"
# else:
#     exec("import config.config_dev as dev")
#     exec("config.update(dev.__dict__)")
#     config.ENV_NAME = "dev"

# CONFIG FEATURE
CONFIG_TEST = r'/home/ShareFolder/factors_ops/config/feature_global_ti0'
configs = load_namespace(CONFIG_TEST)
configs.update(config)

