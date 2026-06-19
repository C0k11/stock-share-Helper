"""配置加载器：把分散的 `yaml.safe_load` 收敛成唯一入口。

加载顺序（后者覆盖前者，深合并）：
    1. 基础 YAML        （默认 configs/default.yaml）
    2. 同名 .local.yaml （如 configs/default.local.yaml，被 .gitignore；放密钥/本机路径）
    3. 环境变量         （QUANTAI__section__key=value，双下划线表示嵌套）
    4. pydantic 校验    -> 返回类型化的 AppConfig（配置错立刻报错）

为什么不用 pydantic-settings：env 覆盖逻辑很简单，手写更透明、也少一个依赖。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .schema import AppConfig

# quantai/config/loader.py -> parents[2] = 仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

ENV_PREFIX = "QUANTAI"
ENV_NESTED_SEP = "__"


def load_yaml(path: Path) -> dict[str, Any]:
    """读取一个 YAML 文件为 dict；空文件返回 {}。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射(dict)，但 {path} 解析为 {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归深合并两个 dict：override 覆盖 base；两边都是 dict 时递归，否则整体替换。"""
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _coerce_scalar(raw: str) -> Any:
    """把环境变量字符串解析成 bool/int/float/None/str（借 YAML 标量规则）。"""
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _env_overrides(
    prefix: str = ENV_PREFIX,
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """从环境变量构造嵌套覆盖 dict。

    例：QUANTAI__api__port=9001 -> {"api": {"port": 9001}}
    """
    env = os.environ if environ is None else environ
    full_prefix = f"{prefix}{ENV_NESTED_SEP}"
    overrides: dict[str, Any] = {}

    for key, value in env.items():
        if not key.startswith(full_prefix):
            continue
        path = key[len(full_prefix):].split(ENV_NESTED_SEP)
        path = [p.lower() for p in path if p]
        if not path:
            continue
        cursor = overrides
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce_scalar(value)

    return overrides


def _local_sibling(path: Path) -> Path:
    """configs/default.yaml -> configs/default.local.yaml"""
    return path.with_name(f"{path.stem}.local{path.suffix}")


def load_config(
    config_path: Optional[Path | str] = None,
    *,
    use_local: bool = True,
    use_env: bool = True,
    environ: Optional[Mapping[str, str]] = None,
) -> AppConfig:
    """加载并校验全局配置，返回类型化 AppConfig。

    Args:
        config_path: 基础 YAML 路径，默认 configs/default.yaml。
        use_local:   是否合并同名 .local.yaml（默认 True）。
        use_env:     是否合并 QUANTAI__* 环境变量（默认 True）。
        environ:     供测试注入的环境字典；None 时用 os.environ。

    Raises:
        FileNotFoundError: 基础配置不存在。
        pydantic.ValidationError: 字段类型/取值/未知键不合法。
    """
    base_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not base_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {base_path}")

    merged: dict[str, Any] = load_yaml(base_path)

    if use_local:
        local_path = _local_sibling(base_path)
        if local_path.exists():
            merged = deep_merge(merged, load_yaml(local_path))

    if use_env:
        env_over = _env_overrides(environ=environ)
        if env_over:
            merged = deep_merge(merged, env_over)

    return AppConfig.model_validate(merged)
