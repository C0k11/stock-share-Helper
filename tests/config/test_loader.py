"""quantai.config.loader 的加载/合并/覆盖行为测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quantai.config.loader import (
    DEFAULT_CONFIG_PATH,
    _env_overrides,
    deep_merge,
    load_config,
)


def test_default_config_loads_and_validates() -> None:
    """仓库自带 configs/default.yaml 必须能加载且通过校验。"""
    cfg = load_config(use_env=False)
    assert cfg.project.name == "QuantAI"
    assert cfg.market.primary == "US"
    assert cfg.backtest.fill_timing == "next_open"
    assert cfg.data.price_source == "yfinance"


def test_deep_merge_recursive() -> None:
    base = {"a": {"x": 1, "y": 2}, "c": 9}
    override = {"a": {"y": 3, "z": 4}, "b": 5}
    assert deep_merge(base, override) == {"a": {"x": 1, "y": 3, "z": 4}, "b": 5, "c": 9}
    # 不可变性：原 base 不被修改
    assert base == {"a": {"x": 1, "y": 2}, "c": 9}


def test_env_overrides_parsing_and_coercion() -> None:
    env = {
        "QUANTAI__api__port": "9001",        # -> int
        "QUANTAI__api__debug": "false",      # -> bool
        "QUANTAI__market__timezone": "UTC",  # -> str
        "PATH": "/usr/bin",                  # 非前缀，忽略
    }
    assert _env_overrides(environ=env) == {
        "api": {"port": 9001, "debug": False},
        "market": {"timezone": "UTC"},
    }


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_local_sibling_override(tmp_path: Path) -> None:
    base = _write(tmp_path / "cfg.yaml", "api:\n  port: 8000\n  debug: true\n")
    _write(tmp_path / "cfg.local.yaml", "api:\n  debug: false\n")
    cfg = load_config(base, use_env=False)
    assert cfg.api.port == 8000     # 来自 base
    assert cfg.api.debug is False   # 被 .local 覆盖


def test_env_override_beats_file(tmp_path: Path) -> None:
    base = _write(tmp_path / "cfg.yaml", "api:\n  port: 8000\n")
    cfg = load_config(base, use_local=False, environ={"QUANTAI__api__port": "9999"})
    assert cfg.api.port == 9999


def test_invalid_value_raises(tmp_path: Path) -> None:
    base = _write(tmp_path / "cfg.yaml", "market:\n  primary: HK\n")  # 违反 US-only
    with pytest.raises(ValidationError):
        load_config(base, use_env=False)


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("definitely_not_here_12345.yaml"))


def test_default_config_path_points_to_repo() -> None:
    assert DEFAULT_CONFIG_PATH.name == "default.yaml"
    assert DEFAULT_CONFIG_PATH.parent.name == "configs"
