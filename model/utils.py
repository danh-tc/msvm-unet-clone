import ast
from typing import Any, Dict, Optional

__all__ = ["register_model", "find_candidates"]


# noinspection PyUnusedLocal
def register_model(*args, **kwargs):
    return args[0] if args and callable(args[0]) else (lambda obj: obj)


def read_source_code(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def parse_decorator(node: ast.AST) -> Dict[str, Any]:
    def get_name(n: ast.AST) -> str:
        if isinstance(n, ast.Call):
            return get_name(n.func)
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            parts = []
            cur = n
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return ""

    def const_eval(n: ast.AST) -> Any:
        # noinspection PyBroadException
        try:
            return ast.literal_eval(n)
        except Exception:
            if isinstance(n, ast.Name):
                return n.id
            elif isinstance(n, ast.Attribute):
                return get_name(n)
            elif isinstance(n, ast.Call):
                return f"{get_name(n)}(...)"
            return None

    info = {
        "name": get_name(node),
        "args": [],
        "kwargs": {},
        "is_call": isinstance(node, ast.Call),
    }

    if isinstance(node, ast.Call):
        info["args"] = [const_eval(e) for e in node.args]
        for kw in node.keywords:
            key = kw.arg if kw.arg else "**"
            info["kwargs"][key] = const_eval(kw.value)

    return info


def parse_alias(deco_info: dict) -> Optional[str]:
    if deco_info["is_call"]:
        alias = deco_info["kwargs"].get("alias", None)
        if alias is not None:
            assert isinstance(alias, str), f"alias must be a string, got {type(alias)}"
            return alias
        num_args = len(deco_info["args"])
        if num_args == 1:
            alias = deco_info["args"][0]
            assert isinstance(alias, str), f"alias must be a string, got {type(alias)}"
            return alias
        elif num_args == 0:
            return None
        raise ValueError("Multiple arguments accepted")
    return None


def find_candidates(filename: str) -> list:
    candidates = []
    tree = ast.parse(read_source_code(filename), filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            for deco in getattr(node, "decorator_list", []):
                deco_info = parse_decorator(deco)
                if not deco_info["name"].endswith("register_model"):
                    continue
                candidates.append(
                    {
                        "name": node.name,
                        "alias": e if (e := parse_alias(deco_info)) else None,
                    }
                )
                break
    return candidates
