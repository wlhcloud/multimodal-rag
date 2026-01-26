from typing import Optional, Tuple
from dots_ocr.dots_parser import DotsOCRParser
from dots_ocr.utils import dict_promptmode_to_prompt


def do_parse(
    input_path: str,
    output: str = "./output",
    prompt: str = "prompt_layout_all_en",
    bbox: Optional[Tuple[int, int, int, int]] = None,
    ip: str = "localhost",
    port: int = 6006,
    model_name: str = "dots_ocr",
    temperature: float = 0.1,
    top_p: float = 1.0,
    dpi: int = 200,
    max_completion_tokens: int = 16384,
    num_thread: int = 16,
    no_fitz_preprocess: bool = False,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    use_hf: bool = False,
):
    """
    dots.ocr 多语言文档布局解析器

    参数:
        input_path (str): 输入PDF/图像文件路径
        output (str): 输出目录 (默认: ./output)
        prompt (str): 用于查询模型的提示词，不同任务使用不同的提示词
        bbox (Optional[Tuple[int, int, int, int]]): 边界框坐标 (x1, y1, x2, y2)
        ip (str): 服务器IP地址 (默认: localhost)
        port (int): 服务器端口 (默认: 8000)
        model_name (str): 模型名称 (默认: model)
        temperature (float): 温度参数 (默认: 0.1)
        top_p (float): 核采样参数 (默认: 1.0)
        dpi (int): DPI设置 (默认: 200)
        max_completion_tokens (int): 最大完成标记数 (默认: 16384)
        num_thread (int): 线程数 (默认: 16)
        no_fitz_preprocess (bool): 是否禁用Fitz预处理 (默认: False)指的是选择是否使用PyMuPDF（fitz）库对图像输入进行特定的预处理操作
        min_pixels (Optional[int]): 最小像素数
        max_pixels (Optional[int]): 最大像素数
        use_hf (bool): 是否使用HuggingFace (默认: False)
    """
    # 获取所有可用的提示模式
    prompts = list(dict_promptmode_to_prompt.keys())

    # 验证prompt参数是否有效
    if prompt not in prompts:
        raise ValueError(f"无效的prompt参数: {prompt}。可选值: {prompts}")

    # 创建DotsOCR解析器实例
    dots_ocr_parser = DotsOCRParser(
        ip=ip,
        port=port,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        num_thread=num_thread,
        dpi=dpi,
        output_dir=output,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_hf=use_hf,
    )

    # 设置Fitz预处理标志
    fitz_preprocess = not no_fitz_preprocess
    if fitz_preprocess:
        print(f"对图像输入使用Fitz预处理，请检查图像像素的变化")

    # 解析文件
    result = dots_ocr_parser.parse_file(
        input_path,
        prompt_mode=prompt,
        bbox=bbox,
        fitz_preprocess=fitz_preprocess,
    )

    return result


if __name__ == "__main__":
    # main()
    # do_parse(input_path='../demo_image1.jpg')
    # do_parse(input_path='../demo_pdf1.pdf', num_thread=32)
    do_parse(input_path="../第一章 Apache Flink 概述.pdf", num_thread=32)
