import os
import time
from http import HTTPStatus
from typing import List, Tuple, Dict, Optional,Any

import dashscope
from dashscope import MultiModalEmbeddingItemImage
from dashscope.embeddings.multimodal_embedding import MultiModalEmbeddingItemBase, MultiModalEmbeddingItemText
from sentence_transformers import SentenceTransformer

from my_llm import gme_st
from utils.env_utils import ALIBABA_API_KEY
from utils.log_utils import log

# =====配置区 =====
DASHSCOPE_MODEL = 'multimodal-embedding-v1'  # 指定使用达摩院多模态嵌入模型
RPM_LIMIT = 120  # 每分钟最多调用次数
WINDOW_SECONDS = 60  # 限流时间窗口(秒), 与RPM_LIMIT配合实现每分钟限流

RETRY_ON_429 = True  # 是否在遇到429(请求过多)状态码时进行重试
MAX_429_RETRIES = 5  # 429状态码的最大重试次数
BASE_BACKOFF = 2.0  # 指数退避算法的基础等待时间(秒)
# 图片最大体积(URL HEAD 检查)，若超过则跳过图片项
MAX_IMAGE_BYTES = 3 * 1024 * 1024
# =====配置区结束 =====


# 全局数据容器，用于存储所有处理后的数据
all_data: List[Dict] = []


class FixedWindowRateLimiter:
    """固定窗口速率限制器类，用于控制API调用频率

    固定窗口算法逻辑：
    1. 将时间划分为固定长度的窗口（如10秒）
    2. 每个窗口内最多允许N次请求
    3. 窗口结束后重置计数，进入新窗口
    """

    def __init__(self, limit: int, window_seconds: int):
        """初始化速率限制器

        Args:
            limit: 时间窗口内允许的最大请求数（必须>0）
            window_seconds: 时间窗口长度(秒)（必须>0）

        Raises:
            ValueError: 当limit或window_seconds非正数时抛出
        """
        # 参数校验，避免无效配置
        if limit <= 0:
            raise ValueError("请求限制数(limit)必须大于0")
        if window_seconds <= 0:
            raise ValueError("时间窗口(window_seconds)必须大于0")

        self.limit = limit
        self.window_seconds = window_seconds
        self.window_start = time.monotonic()  # 使用monotonic避免系统时间修改导致的问题
        self.count = 0  # 当前窗口内的请求计数
        # 加锁保证多线程/多进程安全
        self._lock = self._get_lock()

    def _get_lock(self):
        """获取锁对象，保证线程安全"""
        try:
            import threading
            return threading.Lock()
        except ImportError:
            # 无threading模块时返回空对象（单线程环境）
            class DummyLock:
                def acquire(self): pass

                def release(self): pass

            return DummyLock()

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> bool:
        """尝试获取请求许可，核心限流逻辑

        Args:
            block: 是否阻塞等待（True=阻塞直到获取许可，False=非阻塞）
            timeout: 阻塞等待的超时时间（秒），仅当block=True时有效；None=无限等待

        Returns:
            bool: True=获取许可成功，False=获取失败（非阻塞/超时）

        Raises:
            ValueError: 当timeout为负数时抛出
        """
        if timeout is not None and timeout < 0:
            raise ValueError("超时时间(timeout)不能为负数")

        start_time = time.monotonic()

        while True:
            with self._lock:  # 加锁保证计数安全
                # 1. 检查是否进入新窗口，若是则重置计数和窗口起始时间
                current_time = time.monotonic()
                window_elapsed = current_time - self.window_start

                if window_elapsed >= self.window_seconds:
                    # 进入新窗口，重置状态
                    self.window_start = current_time
                    self.count = 0

                # 2. 检查当前窗口是否还有剩余额度
                if self.count < self.limit:
                    self.count += 1
                    return True

            # 3. 无额度时处理阻塞/超时逻辑
            if not block:
                return False

            # 计算需要等待到下一个窗口的时间
            time_to_next_window = self.window_seconds - (time.monotonic() - self.window_start)
            if time_to_next_window <= 0:
                time_to_next_window = 0.01  # 避免等待0秒导致死循环

            # 检查是否超时
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False
                # 调整等待时间，不超过剩余超时时间
                time_to_next_window = min(time_to_next_window, timeout - elapsed)

            # 等待到下一个窗口
            time.sleep(time_to_next_window)

    def get_remaining(self) -> int:
        """获取当前窗口剩余的请求额度"""
        with self._lock:
            current_time = time.monotonic()
            if current_time - self.window_start >= self.window_seconds:
                return self.limit
            return self.limit - self.count

    def reset(self):
        """手动重置限流状态"""
        with self._lock:
            self.window_start = time.monotonic()
            self.count = 0


limiter = FixedWindowRateLimiter(RPM_LIMIT, WINDOW_SECONDS)

def image_to_base64(img: str) -> tuple[str, str]:
    """将图片转换为base4编码"""
    try:
        import base64, mimetypes
        # 猜测文件MIME类型
        mime = mimetypes.guess_type(img)[0] or "image/png"  # 读取文件并编码为base64
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # 构建data URI格式
        api_img = f"data:{mime};base64,{b64}"
        # store 用原路径或basename 或URL 原值，这里存原字符串
        return api_img, img
    except Exception as e:
        print(f"[图片]本地文件转base64失败:{e}")
        log.exception(e)
    return "", ""


def normalize_image(img: str) -> Tuple[str, str]:
    """规范化图像输入，处理URL和本地文件两种类型

    返回元组(api_image,store_image)
    api_mage 用于向量化;store_image 用于入库;
    若图片无效或超过限制，则返回(""，"")

    Args:
        img:图像路径或URL字符串
    Returns:
        Tuple[str，str]:(用于API的图像数据，用于存储的图像标识)
    """
    if not img:
        return "", ""

    raw = img.strip()  # 去掉首尾空格
    low = raw.lower()  # 转换为小写便于判断

    # URL处理
    if low.startswith("https://") or low.startswith("http://"):
        try:
            import requests
            # 发送HEAD请求图片信息
            head = requests.head(raw, timeout=5, allow_redirects=True)
            if head.status_code == 200:
                # 获取图片大小
                size = int(head.headers.get("Content-Length") or 0)
                if size and size > MAX_IMAGE_BYTES:
                    print(f'[图片] URL 大小 {size} > {MAX_IMAGE_BYTES}, 跳过该图：{raw}')
                    return "", ""
            else:
                print(f'[图片] URL 不可达，status {head.status_code}：{raw}')
                return "", ""
        except Exception as e:
            print(f'[图片] HEAD 检测异常: {e}')

    # 本地文件处理
    if os.path.isfile(raw):
        return image_to_base64(raw)

    # 其他不支持的类型
    return "", ""

def local_gme_one(input_data:List[Dict[str,Any]]) ->Tuple[
    bool, List[float], Optional[int], Optional[float]]:
    """
    调用本地GME模型多模态向量
    :param input_data:
    :return:
    """
    encode_data = []
    for data in input_data:
        if data.__contains__('image'):
            encode_data.append(data.get('image'))
        elif data.__contains__('text'):
            encode_data.append(data.get('text'))

    embedding = gme_st.encode(encode_data,convert_to_tensor=True)
    return True,embedding[0].tolist(),None,None

def call_dashscope_once(input_data: List[Dict]) -> Tuple[
    bool, List[float], Optional[int], Optional[float]]:
    """调用达摩院多模态嵌入AFI一次
    Args:
        input_data:输入数据列表，包含文本或图像数据
    Returns:
        Tuple:(成功标志，嵌入向量，HTTP状态码，重试等待时间)
    """
    # 应用速率限制
    limiter.acquire()

    req_data:List[MultiModalEmbeddingItemBase] = []
    for data in input_data:
        if data.__contains__('image'):
            req_data.append(MultiModalEmbeddingItemImage(data.get('image'),data.get('factor',1)))
        elif data.__contains__('text'):
            req_data.append(MultiModalEmbeddingItemText(data.get('text'),data.get('factor',1)))

    try:
        # 调用达摩院多模态嵌入API
        response = dashscope.MultiModalEmbedding.call(model=DASHSCOPE_MODEL, input=req_data,
                                                      api_key=ALIBABA_API_KEY)
    except Exception as e:
        print(f"调用 DashScope 异常:{e}")
        log.exception(e)
        return False, [], None, None
    # 获取HTTP状态码
    status = getattr(response, "status_code", None)
    retry_after = None

    # 检查是否需要重试等待
    try:
        headers = getattr(response, "headers", None)
        if headers and isinstance(headers, dict):
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra:
                retry_after = float(ra)
    except Exception as e:
        pass
    # 获取API返回的代码和消息
    resp_code = getattr(response, "code", "")
    resp_msg = getattr(response, "message", "")

    if status == HTTPStatus.OK:
        # 提取嵌入向量
        embedding = response.output['embeddings'][0]['embedding']
        return True, embedding, status, retry_after
    else:
        # 处理失败响应
        print(f"请求失败, 状态码:{status}, code: {resp_code},message: {resp_msg}")
        return False, [], status, retry_after


def process_item_with_guard(item: Dict) -> Dict:
    """处理单个数据项(文本或图像)，生成嵌入向量

    mode =‘text':文本项:把content 向量化;
    mode=‘image':图片项:问量化图片

    Args:
        item:原始数据项
    Returns:
        Dict:处理后的数据项，包含嵌入向量
    """
    # 创建原始项的副本以避免修改原数据
    new_item = item.copy()
    raw_content = (new_item.get('text') or '').strip()
    image_raw = (new_item.get("image_path")or '').strip()

    if image_raw:
        img = normalize_image(image_raw)[0]
        input_data = [{'text':raw_content,'factor':1},{'image':img,'factor':1}]
        log.info(f'图片：{image_raw},所对应的描述为{raw_content}')
    else:
        input_data = [{'text':raw_content,'factor':1}]

     # 调用本地模型获取图像嵌入向量
    ok, embedding, status, retry_after = local_gme_one(input_data)
    if ok:
         new_item['dense'] = embedding  # 成功时添加嵌入向量
    else:
        new_item['dense'] = []  # 失败时设置为空数组
    return new_item

if __name__ == '__main__':
    pass