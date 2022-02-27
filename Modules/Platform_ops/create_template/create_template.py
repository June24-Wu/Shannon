import os
import re
import sys

TEMPLATE_PATH = r'/home/ShareFolder/lgc/Modules/Platform/alpha_template'


def _copy(new_cls_name, new_fun_name, output_path, mode):
    cls_name = ''
    fun_name = ''

    pattern_cls = re.compile(r'class\s(\S+)[(]Alpha[)]')
    pattern_fun = re.compile(r".*def\s(\S+)[(]self, start_time, end_time, output=None, mode='dev'[)]")

    ori_template = os.path.join(TEMPLATE_PATH, 'source/template.py')
    if mode == 0:
        new_template = os.path.join(output_path, f'source/{new_cls_name}.py')
    else:
        new_template = os.path.join(output_path, f'{new_cls_name}.py')

    new_cls_name = new_cls_name.capitalize()
    new_fun_name = new_fun_name.lower()

    ori_f = open(ori_template, 'r', encoding='utf-8')
    new_f = open(new_template, 'w', encoding='utf-8')

    for line in ori_f.readlines():
        m_cls = pattern_cls.match(line)
        m_fun = pattern_fun.match(line)

        if m_cls:
            cls_name = m_cls.group(1)
            line = line.replace(cls_name, new_cls_name)

        elif cls_name and cls_name in line:
            line = line.replace(cls_name, new_cls_name)

        elif m_fun:
            fun_name = m_fun.group(1)
            line = line.replace(fun_name, new_fun_name)

        elif fun_name:
            line = line.replace(fun_name, new_fun_name)

        new_f.write(line)

    ori_f.close()
    new_f.close()


def copy_dir(cls_name, fun_name, output_path):
    output_path = os.path.abspath(output_path)
    output_path = os.path.join(output_path, fun_name)

    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        os.system(f'cp -r {TEMPLATE_PATH}/* {output_path}')
    except Exception as e:
        print(e)
    else:
        _copy(cls_name, fun_name, output_path, mode=0)


def copy_file(cls_name, fun_name, output_path):
    output_path = os.path.abspath(output_path)
    _copy(cls_name, fun_name, output_path=output_path, mode=1)


def main():
    args = sys.argv

    if args[1] == '--help':
        print("""
        Please input class name, function name, mode, and output path.
        The class name must start with a capital letter and cannot start with a number or other symbol.
        The method name must start with a lowercase letter and cannot start with a number or other symbol.
        Mode project means copy directory, file mean copy file. 
        The output path defaults to None, means current directory.""")
        return

    if len(args) <= 3 or len(args) >= 5:
        print('arg error, you can type --help to view details.')
        return

    cls_name = args[2]
    fun_name = args[2]
    mode = args[1]

    if len(args) == 4:
        output_path = args[3]
        if not output_path:
            output_path = '../tools'
    else:
        output_path = '../tools'

    if mode == 'project':
        copy_dir(cls_name, fun_name, output_path)
    elif mode == 'file':
        copy_file(cls_name, fun_name, output_path)
    else:
        print('arg error, you can type --help to view details.')


if __name__ == '__main__':
    main()
