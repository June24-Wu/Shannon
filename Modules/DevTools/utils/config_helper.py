class Config(dict):
    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)
