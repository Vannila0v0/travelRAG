import json


def convert_json_to_md(input_file, output_file):
    try:
        # 1. 读取 JSON 数据
        # 确保使用 utf-8 编码以支持中文
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 如果 JSON 根对象是字典而不是列表，将其包装进列表
        if isinstance(data, dict):
            data = [data]

        # 2. 准备写入 Markdown 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                prompt = item.get("prompt", "")
                chosen = item.get("chosen", "")

                # 3. 按照格式组合字符串
                # #### 标题
                # 内容
                # (空行)
                md_content = f"#### {prompt}\n{chosen}\n\n"

                f.write(md_content)

        print(f"成功！已将数据转换并保存至: {output_file}")

    except Exception as e:
        print(f"处理过程中出错: {e}")


# 执行转换
# 假设你的文件名叫 data.json
convert_json_to_md('data/tourism_dpo.json', 'data/tourism_dpo.md')