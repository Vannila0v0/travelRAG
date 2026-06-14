import re
import string

def normalize_answer(s: str) -> str:
    """
    标准化答案文本：
    1. 转换为小写
    2. 移除标点符号
    3. 移除多余的空白字符
    主要用于中文和英文的字符串基础比对。
    """
    if not s:
        return ""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        # 移除英文标点
        exclude = set(string.punctuation)
        # 移除常见的中文标点
        zh_punctuation = set("！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.")
        exclude.update(zh_punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))