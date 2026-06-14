from transformers import AutoModel, AutoTokenizer


model_name = "Qwen/Qwen2.5-7B-Instruct"
cache_dir = r"E:\MyOwnProj\local-rag-lab\cache\model"
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print("ok")