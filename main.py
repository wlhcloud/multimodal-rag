import os
import gradio as gr
from dots_ocr.parser import do_parse
from utils.common_utils import (
    delete_directory_if_non_empty,
    get_filename,
    get_sorted_md_files,
)
from utils.log_utils import log

# md存储的临时模型
base_md_dir = r"./output"
OCR_IP = "127.0.0.1"
OCR_PORT = 6007


class ProcessorAPP:

    def __init__(self):
        self.pdf_path = None
        self.md_path = None
        self.md_files = None
        self.md_dir = None
        self.file_contents = None

    def upload_pdf(self, pdf_file):
        """处理PDF文件上传"""
        log.info(f"上传pdf文件：{pdf_file}")
        self.pdf_path = pdf_file if pdf_file else None
        if self.pdf_path:
            return [
                f"PDF已上传：{os.path.basename(self.pdf_path)}",
                gr.Button(interactive=True),
            ]
        else:
            return [
                f"上传失败！请重新上传{os.path.basename(self.pdf_path)}",
                gr.Button(interactive=True),
            ]

    def parse_pdf(self):
        """解析pdf文件，变成多个md文件"""
        md_files_dir = os.path.join(base_md_dir, get_filename(self.pdf_path, False))
        # 删除
        delete_directory_if_non_empty(md_files_dir)
        do_parse(
            input_path=self.pdf_path,
            num_thread=32,
            no_fitz_preprocess=True,
            ip=OCR_IP,
            port=OCR_PORT,
        )
        if os.path.isdir(md_files_dir):
            self.md_dir = md_files_dir
            log.info(f"MD files directory created: {md_files_dir}")
            # 要先排序
            self.md_files = get_sorted_md_files(self.md_dir)
            self.md_files = [f for f in self.md_files if "nohf" in f]
            log.info(f"PDF已解析，生成的MD文件列表：{self.md_files}")

            self.file_contents = {}
            for f in self.md_files:
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        self.file_contents[f] = file.read()
                except Exception as e:
                    log.error(f"读取文件 {f} 时出错：{e}")
                    self.file_contents[f] = f"读取文件内容时出错：{e}"
            file_names = [os.path.basename(f) for f in self.md_files]
            return [
                f"PDF解析成功！共{len(self.md_files)} 个MD文件",
                gr.Dropdown(
                    choices=file_names,
                    label="MD文件列表",
                    interactive=True,
                ),
                gr.Button(interactive=False),  # parse_btn
                gr.update(interactive=True),  # save_btn
            ]
        else:
            return [
                "PDF解析失败！",
                gr.Dropdown(
                    choices=[],
                    label="MD文件列表",
                    interactive=False,
                ),
                gr.Button(interactive=True),  # parse_btn
                gr.update(interactive=False),  # save_btn
            ]

    def select_md_file(self, selected_file):
        """选择一个md文件，并展示他的内容"""
        log.info(f"选择文件：{selected_file}")
        if selected_file:
            show_file = None
            for f in self.md_files:
                if os.path.basename(f) == selected_file:
                    show_file = f
                    break

            if show_file and show_file in self.file_contents:
                return self.file_contents[show_file]
            else:
                return f"没有找到这个文件！"
        else:
            return "文件内容加载失败，选择文件不对!"

    def create_interface(self):
        """
        创建一个构建多模态的知识库的界面
        """
        with gr.Blocks() as app:
            gr.Markdown("## PDF解析与知识库存储和构建")

            with gr.Row():
                pdf_upload = gr.File(label="上传PDF文件")
                parse_btn = gr.Button("解析PDF", variant="primary")

            status = gr.Textbox(label="状态", value="等待操作....", interactive=False)

            with gr.Row():
                # 文件列表
                file_dropdown = gr.Dropdown(
                    choices=[], label="MD文件列表", interactive=False
                )
                # MD文件中的内容
                content = gr.Textbox(label="文件内容", lines=20, interactive=False)

            save_btn = gr.Button("存入知识库", variant="secondary", interactive=False)

            # 绑定按钮事件
            pdf_upload.change(
                fn=self.upload_pdf, inputs=pdf_upload, outputs=[status, parse_btn]
            )
            parse_btn.click(
                fn=self.parse_pdf, outputs=[status, file_dropdown, parse_btn, save_btn]
            )
            file_dropdown.change(
                fn=self.select_md_file, inputs=file_dropdown, outputs=content
            )
        return app


if __name__ == "__main__":
    app = ProcessorAPP()
    app.create_interface().launch()
