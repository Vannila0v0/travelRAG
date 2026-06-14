#!/usr/bin/env python3
"""
benchmark.py – quick-n-dirty load-tester for your self-hosted LLM.

Metrics
-------
1. max_context_tokens       : 最大可接受上下文长度 (tokens)
2. max_cot_tokens           : “思维链”最大长度 (prompt 要求模型输出思维链时，模型能给出的最大 tokens)
3. max_output_tokens_cfg    : 可配置的最大输出长度 (max_tokens 参数支持的上限)
4. max_output_tokens_default: 不传 max_tokens 时模型默认能返回的最大长度
5. throughput_rps           : 吞吐量 – 单并发持续 60 s 请求的平均 RPS
6. generation_tps           : Token 生成速度 (tokens / second)
7. max_concurrency          : 不报 429/5xx 时可同时跑通的最大并发

Notes
-----
* 预设使用 OpenAI-compatible /v1/chat/completions 端点；如命名不同请自行替换。
* 用 tiktoken 粗略估计 prompt token 数，若您的模型有专用 tokenizer 可替换。
* 为了节省测试成本，脚本里所有 prompt 都很短；真正极限探测时会自动递增规模。
"""

import asyncio, time, os, math, statistics, sys
import httpx
import tiktoken

API_KEY   = os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", "YOUR_API_KEY"))
BASE_URL  = os.getenv("DEEPSEEK_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
MODEL     = os.getenv("DEEPSEEK_MODEL", os.getenv("LLM_MODEL", "deepseek-chat"))

HEADERS   = {"Authorization": f"Bearer {API_KEY}"}
ENDPOINT  = f"{BASE_URL}/v1/chat/completions"
ENC       = tiktoken.get_encoding("cl100k_base")  # OpenAI 兼容 tokenizer，必要时替换

SYSTEM    = {"role": "system", "content": "You are a helpful assistant."}
USER_BASE = "请用简体中文回答："

###############################################################################
# helpers
###############################################################################
async def chat(client, msg, max_tokens=None):
    payload = {
        "model": MODEL,
        "messages": [SYSTEM, {"role": "user", "content": msg}],
        "stream": False
    }
    if max_tokens: payload["max_tokens"] = max_tokens
    r = await client.post(ENDPOINT, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"], data["usage"]

def token_len(text):                    # 粗略 token 计数
    return len(ENC.encode(text))

###############################################################################
# 1) & 2) 探测最大上下文 / 最大思维链长度
###############################################################################
async def find_limit(client, base_prompt, field="prompt"):
    step   = 1024          # 每次递增 tokens
    max_tk = step
    while True:
        if field == "prompt":
            prompt = base_prompt * math.ceil(max_tk / token_len(base_prompt))
            try:
                await chat(client, prompt[:max_tk*2])  # 再保险一点
                max_tk += step
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 413):  # 长度超限
                    return max_tk - step
                raise
        else:  # measure output limit
            try:
                _, usage = await chat(client, base_prompt, max_tokens=max_tk)
                if usage["completion_tokens"] < max_tk:
                    max_tk += step
                else:
                    return max_tk
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 413):
                    return max_tk - step
                raise

###############################################################################
# 3) 可配置 & 默认输出长度
###############################################################################
async def output_limits(client):
    # configurable
    max_cfg = await find_limit(client, USER_BASE + "输出任意内容直到被截断。", field="output")
    # default
    content, usage = await chat(client, USER_BASE + "请输出尽可能长的文本，以测试默认最大长度。")
    return max_cfg, usage["completion_tokens"]

###############################################################################
# 4) 吞吐量 (RPS, 单并发 60 s)
###############################################################################
async def throughput_1min(client):
    stop = time.time() + 60
    cnt  = 0
    while time.time() < stop:
        await chat(client, USER_BASE + "返回 OK")
        cnt += 1
    return cnt / 60

###############################################################################
# 5) Token 生成速度 (tokens/sec)
###############################################################################
async def token_speed(client):
    t0 = time.time()
    content, usage = await chat(client, USER_BASE + "请输出不少于 256 个汉字。", max_tokens=512)
    dt = time.time() - t0
    return usage["completion_tokens"] / dt

###############################################################################
# 6) 最大并发
###############################################################################
async def max_concurrency(client):
    async def worker(idx):
        await chat(client, f"{USER_BASE}并发测试 #{idx}")
    low, high = 1, 128         # 自行调整上限
    while low < high:
        mid   = (low + high + 1) // 2
        tasks = [worker(i) for i in range(mid)]
        try:
            await asyncio.gather(*tasks)
            low = mid
        except (httpx.HTTPStatusError, httpx.ReadTimeout):
            high = mid - 1
    return low

###############################################################################
async def main():
    async with httpx.AsyncClient(http2=True) as client:
        ctx_max     = await find_limit(client, USER_BASE + "填充")
        cot_max     = await find_limit(client,
                                       USER_BASE + "你需要一步步展示思考链，直到被截断。",
                                       field="output")
        out_cfg, out_def = await output_limits(client)
        rps         = await throughput_1min(client)
        tps         = await token_speed(client)
        conc_max    = await max_concurrency(client)

    print("\n=== Benchmark Report ===")
    print(f"最大上下文长度             : {ctx_max} tokens")
    print(f"最大思维链内容长度         : {cot_max} tokens")
    print(f"可配置最大输出长度         : {out_cfg} tokens")
    print(f"默认最大输出长度           : {out_def} tokens")
    print(f"吞吐量 (1 并发, 60s 平均)  : {rps:.2f} requests / sec")
    print(f"token 生成速度             : {tps:.2f} tokens / sec")
    print(f"最大并发 (无错误)          : {conc_max} simultaneous requests")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
