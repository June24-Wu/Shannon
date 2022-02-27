# !/bin/bash

# /bin/python3改成所在服务器上的python3绝对路径
# /home/liuhuan/code/eval/create_template.py 改成create_template保存的绝对路径
# chmod 授予执行权限，然后可以使用ln -s create.sh /bin/create 即可全局调用create脚本
/usr/bin/python3 /home/ShareFolder/lgc/Modules/Platform_ops/platform_ops/create_template/create_template.py "$1" "$2" "$3"
