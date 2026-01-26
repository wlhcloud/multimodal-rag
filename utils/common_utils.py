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


def get_surrounding_text_content(data_list, index):
    """
    获取指定图片字典的前后有效文本内容（跳过连续的图片，找到第一个非图片文本）

    :param data_list: 包含字典的列表，每个字典都有'text'和'image_path' 键
                      - image_path非空/非None → 图片元素
                      - image_path为空/None → 文本元素
    :param index: 当前图片字典在列表中的索引
    :return: 一个元组(prev_text, next_text)
        prev_text: 向前遍历找到的第一个非图片文本内容（无则返回空字符串）
        next_text: 向后遍历找到的第一个非图片文本内容（无则返回空字符串）
    :raises:
        TypeError: 索引非整数时抛出
        IndexError: 索引超出列表范围时抛出
        KeyError: 字典缺少'text'/'image_path'键时抛出
    """
    # 1. 基础参数校验
    if not isinstance(index, int):
        raise TypeError(f"索引必须是整数类型，当前传入：{type(index)}")
    if len(data_list) == 0:
        raise IndexError("数据列表为空，无法获取内容")
    if index < 0 or index >= len(data_list):
        raise IndexError(f"索引{index}超出列表范围（列表长度：{len(data_list)}）")

    # 2. 定义辅助函数：判断是否为图片元素
    def is_image_item(item):
        """判断字典是否为图片元素（image_path非空/非None）"""
        if "image_path" not in item:
            raise KeyError("字典缺少'image_path'键")
        # 图片判定：image_path非空、非None、非空字符串
        image_path = item["image_path"]
        return image_path is not None and image_path != ""

    # 3. 向前遍历：找第一个非图片的文本（跳过连续图片）
    prev_text = ""
    # 从index-1开始向前遍历，直到列表开头
    for i in range(index - 1, -1, -1):
        current_item = data_list[i]
        # 不是图片元素 → 提取文本并终止遍历
        if not is_image_item(current_item):
            if "text" not in current_item:
                raise KeyError(f"列表索引{i}的字典缺少'text'键")
            prev_text = current_item["text"].strip()
            break  # 找到第一个有效文本，停止向前遍历

    # 4. 向后遍历：找第一个非图片的文本（跳过连续图片）
    next_text = ""
    # 从index+1开始向后遍历，直到列表末尾
    for i in range(index + 1, len(data_list)):
        current_item = data_list[i]
        # 不是图片元素 → 提取文本并终止遍历
        if not is_image_item(current_item):
            if "text" not in current_item:
                raise KeyError(f"列表索引{i}的字典缺少'text'键")
            next_text = current_item["text"].strip()
            break  # 找到第一个有效文本，停止向后遍历

    return prev_text, next_text