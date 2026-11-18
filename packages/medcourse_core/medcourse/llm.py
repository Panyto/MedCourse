"""本机模型服务的客户端。守则 P1：真实文书只能送 127.0.0.1，这里写死，别的地址直接报错。key 从环境变量读，不打印、不写文件。"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

LOOPBACK = {"127.0.0.1", "localhost", "::1"}

# 守则 v1.4 第 2.3 节：ys 上的 vLLM 拿模型自带的 generation_config 补默认值（14B、8B 对话版是 top_p 0.95、top_k 20，基座版没有），
# 所以每次请求都写明全部采样参数。取值同关1：抽取同 G1-D、G1-E 的评测调用（贪心、不截断），写作同 G1-B 的生成调用。
EXTRACT = {"temperature": 0.0, "top_p": 1.0, "top_k": -1}
WRITE = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
FIXED = {"min_p": 0.0, "repetition_penalty": 1.0}


class LLMError(Exception):
    """只带类型代码，不带回复内容。"""


class Client:
    def __init__(self, base_url, model, raw=False, timeout=600, retries=2):
        host = urllib.parse.urlsplit(base_url).hostname
        if host not in LOOPBACK:
            raise LLMError("not_loopback")
        self.base_url, self.model, self.raw = base_url.rstrip("/"), model, raw
        self.timeout, self.retries = timeout, retries
        self.key = os.environ.get("VLLM_API_KEY", "EMPTY")

    def _post(self, path, body):
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        err = "unknown"
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err = f"http_{e.code}"
                if e.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                err = type(e).__name__
            time.sleep(2 * (attempt + 1))
        raise LLMError(err)

    def generate(self, *, sampling, seed, messages=None, prompt=None, max_tokens=512, schema=None, stop=None):
        """sampling 用 EXTRACT 或 WRITE，seed 必须给。返回 {"text","finish","prompt_tokens","completion_tokens","seconds"}。"""
        if set(sampling) != set(WRITE) or seed is None:
            raise LLMError("sampling_not_explicit")
        body = {"model": self.model, "max_tokens": max_tokens, "seed": seed, **sampling, **FIXED}
        if stop:
            body["stop"] = stop
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "mc", "schema": schema}}
        t0 = time.perf_counter()
        if self.raw:
            body["prompt"] = prompt if prompt is not None else "\n\n".join(m["content"] for m in messages)
            data = self._post("/completions", body)
            text = data["choices"][0].get("text") or ""
        else:
            body["messages"] = messages if messages is not None else [{"role": "user", "content": prompt}]
            body["chat_template_kwargs"] = {"enable_thinking": False}
            data = self._post("/chat/completions", body)
            text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return {"text": text, "finish": data["choices"][0].get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens") or 0,
                "completion_tokens": usage.get("completion_tokens") or 0, "seconds": time.perf_counter() - t0}
