from tools.encrypt import build_so
import os

path_in_list = ['/home/liguichuan/Desktop/Project/release/Platform/feature/ft.py',
                '/home/liguichuan/Desktop/Project/release/Platform/backtest/bt.py']
for path_in in path_in_list[1:2]:
    path_out = os.path.dirname(path_in)
    build_so(path_in, path_out)
