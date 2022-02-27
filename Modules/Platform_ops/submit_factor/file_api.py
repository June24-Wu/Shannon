import os

GITLAB_SAVE_PATH = r'/home/ShareFolder/factors_ops'
# GITLAB_SAVE_PATH = r'/home/qumingming'
from Platform_ops.platform_ops.tools.get_gitlab_file import GitlabDownload


def clear(path):
    if os.path.isdir(path):
        for cur_dir, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    py_file_path = os.path.join(cur_dir, file)
                    os.remove(py_file_path)
    else:
        os.remove(path)


def get_file_api(project_id, filename):
    # token = 'REs7Gm2JM2Jfa_fVdfMj'
    token = 'Ntn9HL9AUdZDwsSp9EY6'
    branch = 'master'
    # python file save path or python file directory
    file_path = os.path.join(GITLAB_SAVE_PATH, filename)
    if os.path.isdir(file_path):
        os_file_path = os.path.join(GITLAB_SAVE_PATH, filename, 'run.so')
    else:
        os_file_name = filename.replace('.py', '.so')
        os_file_path = os.path.join(GITLAB_SAVE_PATH, os_file_name)
    try:
        gitlab = GitlabDownload(project_id, token)
        file = gitlab.get_files(filename, branch, GITLAB_SAVE_PATH)
        python_exec = r'/home/ShareFolder/python-venv/platform-venv/bin/python'
        encrypt_file = r'/home/ShareFolder/lgc/Modules/Platform_ops/platform_ops/tools/encrypt.py'
        cmd = f"{python_exec} {encrypt_file} {file_path} {GITLAB_SAVE_PATH}"
        os.system(cmd)
        clear(file_path)
        return os_file_path

    except Exception as e:
        print(e)
        clear(file_path)
        return
