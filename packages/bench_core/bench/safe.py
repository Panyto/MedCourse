"""输出闸门（守则 1.2）。评测台面向终端和文件的一切输出都走这里：只出计数、标签、位置，不出正文。

异常一律只带类型名和代码位置（文件、行号），不带 message：message 里可能夹着病历字符串。
"""
import json
import os
import sys
import traceback

_CJK_RUN = 15


class SafeError(Exception):
    """只带位置的错误。where 是 row、住院号、文书序号一类，不放内容。"""

    def __init__(self, code, **where):
        super().__init__(code)
        self.code, self.where = code, where


def _long_cjk(s):
    run = 0
    for ch in s:
        run = run + 1 if "一" <= ch <= "鿿" else 0
        if run >= _CJK_RUN:
            return True
    return False


def log(tag, **kw):
    """打一行 JSON 日志。值里只允许数字、布尔、短标签；连续 15 个汉字以上的字符串整条屏蔽。"""
    clean = {}
    for k, v in kw.items():
        if isinstance(v, str) and (_long_cjk(v) or len(v) > 200):
            clean[k] = "<masked>"
        else:
            clean[k] = v
    sys.stderr.write(json.dumps({"tag": tag, **clean}, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    sys.stderr.flush()


def where_exc():
    """当前异常的类型和代码位置，不含 message。"""
    et, _, tb = sys.exc_info()
    frames = traceback.extract_tb(tb)
    last = frames[-1] if frames else None
    return {"exc": et.__name__ if et else "", "file": os.path.basename(last.filename) if last else "",
            "line": last.lineno if last else 0}


def guarded(fn):
    """入口包一层：异常只留类型和位置，退出码 2。"""
    try:
        return fn()
    except SafeError as e:
        log("error", code=e.code, **{k: str(v) for k, v in e.where.items()})
        sys.exit(2)
    except Exception:
        log("error", **where_exc())
        sys.exit(2)


def numeric_only(row):
    """落盘前检查：一行读数只许有数字、布尔、None 和规定的标识字段。"""
    for k, v in row.items():
        if k in ("row", "id", "metric", "reader", "note", "code"):
            continue
        if not (v is None or isinstance(v, (int, float, bool))):
            raise SafeError("non_numeric_field", field=k)
    return row
