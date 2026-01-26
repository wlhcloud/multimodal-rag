import os
from pathlib import Path
import re
import shutil
from typing import List


def get_filename(file_path, with_extension=True):
    """
    获取文件名

    :param file_path: 文件绝对路径
    :param with_extension: 是否包含扩展名
    :return: 文件名
    """
    if with_extension:
        return os.path.basename(file_path)
    else:
        return Path(file_path).stem


def get_sorted_md_files(input_dir: str) -> List[str]:
    """
    按照页号，把所有的md文件排序。（xx_0.md,xx_1md）
    获取指定目录下所有.md 文件，并按照_page_X 中的X 数值排序
    """
    # 获取所有 .md文件
    md_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    # 定义排序key 函数:提取_page_后的数字
    def sort_key(file_path: str) -> int | float:
        filename = os.path.basename(file_path)
        match = re.search(r"_page_(\d+)", filename)
        if match:
            return int(match.group(1))
        else:
            # 如果没有找到数字，则排在最后
            return float("inf")

    # 按照数字排序
    sorted_files = sorted(md_files, key=sort_key)
    return sorted_files


def delete_directory_if_non_empty(dir_path):
    """
    删除指定目录(如果该目录存在且非空)
    :param dir_path: 要检查并可能删除的目录路径
    """

    # 检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"目录‘{dir_path}’不存在，无需删除。")
        return False
    # 确认路径是一个目录
    if not os.path.isdir(dir_path):
        print(f"'{dir_path}'不是一个目录。")
        return False

    # 删除目录及其内容
    import shutil

    shutil.rmtree(dir_path)
    print(f"目录'{dir_path}'已删除。")
    return True
