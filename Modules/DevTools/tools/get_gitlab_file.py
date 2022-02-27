import os
import sys
import gitlab


class GitlabDownload(object):
    def __init__(self, pro_id, token):
        """
        :param pro_id: Sign on gitlab, click Projects--Your project, choose and click the project you want,
        you will find the project ID under the project name.
        :param token: Search by yourself how gitlab generates token.
        """
        self.pro_id = pro_id
        self.token = token

        if sys.platform.lower() == 'linux':
            self.url = "http://172.16.2.6:60005"
        else:
            self.url = "http://114.80.222.242:60005"

    def _login(self):
        gl = gitlab.Gitlab(self.url, self.token)
        return gl

    def _get_project(self, pro_id):
        gl = self._login()
        projects = gl.projects.get(pro_id)
        return projects

    def is_file_exist(self, git_file, git_branch):
        try:
            projects = self._get_project(self.pro_id)
            projects.files.get(file_path=f'{git_file}', ref=f'{git_branch}')
            return True
        except Exception as e:
            print(e)
            return False

    def _get_content(self, git_file, git_branch, loc_folder):
        """
        :param git_branch: gitlab branch
        :param git_file: the file path.
        eg: http://114.80.222.242:60005/dev/platform/-/blob/master/backend/requirements.txt
        the file path is 'backend/requirements.txt'
        :param loc_folder: The path of the file you want to save.
        """

        loc_path = os.path.join(loc_folder, git_file)
        folder = os.path.dirname(loc_path)

        if not os.path.exists(folder):
            os.makedirs(folder)

        try:
            projects = self._get_project(self.pro_id)
            f = projects.files.get(file_path=f'{git_file}', ref=f'{git_branch}')
            content = f.decode()

            with open(f'{loc_path}', 'wb') as code:
                code.write(content)

        except Exception as e:
            print(f'Fail to get file: {git_file}, error message:{e}')
        else:
            print(f'Success, download file "{loc_folder}" complete.')

    def get_files(self, target_file, branch, loc_folder):
        project = self._get_project(self.pro_id)

        file_name = os.path.basename(target_file)
        if '.' in file_name and file_name[0] != '.':
            # file
            self._get_content(target_file, branch, loc_folder)

        else:
            # folder
            res = project.repository_tree(path=target_file, recursive=True, as_list=True, per_page=10000, all=True)
            for i in res:
                path = i.get('path')
                if not path.startswith(target_file):
                    continue
                if '__pycache__' in path or '.idea' in path or '.' not in path:
                    continue
                self._get_content(i.get('path'), branch, loc_folder)


if __name__ == '__main__':
    token = 'R7rPYbyvcaBjNy2gadhE'
    output_path = r'/home/ShareFolder/lgc/data/test/lh/test'
    gd = GitlabDownload(10, token)
    gd.get_files('backend', 'master', output_path)
