from utils.config_helper import Config
from Research.utils.namespace import load_namespace
import os

env = os.getenv("ENV")
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

telephone_dict = {'liguichuan': '13916392907', 'sunyuting': '18813135183', 'fangxin': '13707665666',
                  'liangbin': '18601626312', 'xushengqiang': '13120929850', 'sunrui': '13552945080'}

# if env == "prod":
#     exec("import config.config_prod as prod")
#     if os.path.exists("./config/config_dev.py"):
#         exec("config.update(prod.__dict__)")
#     config.ENV_NAME = "prod"
# elif env == "test":
#     exec("import config.config_test as test")
#     if os.path.exists("./config/config_dev.py"):
#         exec("config.update(test.__dict__)")
#     config.ENV_NAME = "test"
# else:
#     exec("import config.config_dev as dev")
#     if os.path.exists("./config/config_dev.py"):
#         exec("config.update(dev.__dict__)")
#     config.ENV_NAME = "dev"

# CONFIG CORR
CONFIG_CORR = r'/home/ShareFolder/factors_ops/config/corr_daily_ti0.config'
configs_corr = load_namespace(CONFIG_CORR)
configs_corr.update(config)

