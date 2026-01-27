import os
from typing import List, Dict

from dots_ocr.parser import do_parse
from utils.env_utils import DOTS_OCR_PORT, DOTS_OCR_IP
from PIL import Image
import json


def parse_pdf_one(filepath: str):
    result = do_parse(
        input_path=filepath,
        num_thread=32,
        no_fitz_preprocess=True,
        ip=DOTS_OCR_IP,
        port=DOTS_OCR_PORT,
    )
    return result


def _convert_parse(results: List[Dict]):
    """
    Processes using the high-level API parse_pdf from DotsOCRParser
    """
    # Create a temporary session directory
    try:
        # Parse the results
        if not results:
            raise ValueError("No results returned from parser")

        # Handle multi-page results
        parsed_results = []
        all_md_content = []
        all_cells_data = []

        for i, result in enumerate(results):
            page_result = {
                "page_no": result.get("page_no", i),
                "layout_image": None,
                "cells_data": None,
                "md_content": None,
                "filtered": False,
            }

            # Read the layout image
            if "layout_image_path" in result and os.path.exists(
                    result["layout_image_path"]
            ):
                page_result["layout_image"] = Image.open(
                    result["layout_image_path"]
                )

            # Read the JSON data
            if "layout_info_path" in result and os.path.exists(
                    result["layout_info_path"]
            ):
                with open(result["layout_info_path"], "r", encoding="utf-8") as f:
                    page_result["cells_data"] = json.load(f)
                    all_cells_data.extend(page_result["cells_data"])

            # Read the Markdown content
            if "md_content_path" in result and os.path.exists(
                    result["md_content_path"]
            ):
                with open(result["md_content_path"], "r", encoding="utf-8") as f:
                    page_content = f.read()
                    page_result["md_content"] = page_content
                    all_md_content.append(page_content)
            page_result["filtered"] = result.get("filtered", False)
            parsed_results.append(page_result)

        combined_md = "\n\n---\n\n".join(all_md_content) if all_md_content else ""

        md_output_path = "combined_document.md"
        with open(md_output_path, "w", encoding="utf-8") as f:
            f.write(combined_md)
        return {
            "parsed_results": parsed_results,
            "combined_md_content": combined_md,
            "combined_cells_data": all_cells_data,
            "total_pages": len(results),
        }
    except Exception as e:
        raise e


def parse_batch_pdf(dir: str):
    """解析pdf文件，变成多个md文件"""

    for file in os.listdir(dir):
        print(file)

    # result = parse_pdf_one("")

    # _convert_parse()

if __name__ == '__main__':
    parse_batch_pdf('/home/gybwg/ai-project/')
