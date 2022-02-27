from Research.utils.namespace import load_namespace
# from Platform_ops.config.config import CONFIG_PATH
# from Platform_ops.config.config import TEST_FACTORS_DAILY_TI0


class Config(dict):
    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

#
# def get_global_configs():
#     print('Loading the configuration from ' + CONFIG_PATH)
#     configs = load_namespace(CONFIG_PATH)
#     return configs, TEST_FACTORS_DAILY_TI0
