import os
import shutil
import sys
from distutils.core import setup
from Cython.Build import cythonize


def _build_so(input_path, output_path):
    """
    :param input_path: The path is dir or python file.
    :param output_path: The path that the .os file you want to save.
    :return:
    """
    if not os.path.exists(input_path):
        print("path not exists.")
        return

    input_path = os.path.abspath(input_path)

    if os.path.isdir(input_path):
        parent_input_path = os.path.dirname(input_path)

        for cur_dir, dirs, files in os.walk(input_path):
            for file in files:
                cur_path = os.path.join(cur_dir, file)
                if '.git' in cur_path or '.idea' in cur_path or 'build' in cur_path:
                    continue

                elif not cur_path.endswith('.py'):
                    continue

                elif 'setup.py' in cur_path:
                    continue

                build_path = cur_path.replace(parent_input_path, output_path)

                folder = os.path.dirname(build_path)

                if not os.path.exists(folder):
                    os.makedirs(folder)

                setup(ext_modules=cythonize([cur_path], language_level=3), script_args=["build_ext", "-b", folder])

    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        setup(ext_modules=cythonize([input_path], language_level=3), script_args=["build_ext", "-b", output_path])


def clear(input_path, output_path):
    if os.path.isfile(input_path):
        target_file_dir = os.path.dirname(input_path)
    else:
        target_file_dir = input_path

    skip_path = os.path.join(os.path.abspath(target_file_dir), 'build')
    build_path = os.path.abspath(output_path)

    cur_path = os.getcwd()

    if not build_path == skip_path:
        shutil.rmtree(os.path.join(cur_path, 'build'))

    for cur_dir, dirs, files in os.walk(target_file_dir):
        for file in files:
            if file.endswith('.c'):
                cur_path = os.path.join(cur_dir, file)
                os.remove(cur_path)

    for cur_dir, dirs, files in os.walk(output_path):
        if 'temp.' in cur_dir:
            shutil.rmtree(cur_dir)

        for file in files:
            if file.endswith('.so'):
                filename = file.split('.')[0]
                old_so_path = os.path.join(cur_dir, file)
                new_so_path = os.path.join(cur_dir, f'{filename}.so')
                os.rename(old_so_path, new_so_path)

            if file.endswith('.c'):
                c_file_path = os.path.join(cur_dir, file)
                os.remove(c_file_path)


def build_so(input_path, output_path):
    _build_so(input_path, output_path)
    clear(input_path, output_path)


if __name__ == '__main__':
    args = sys.argv
    in_path = args[1]
    out_path = args[2]
    build_so(in_path, out_path)
