"""loader.py 单测：YAML/CSV 加载、校验、lot 语义、隐私示例文件有效性。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quantai.portfolio import Portfolio, Position, load_portfolio

_YAML = """
cash: 1000.5
positions:
  - symbol: nvda
    shares: 10
    cost_basis: 450.5
    open_date: 2025-11-03
  - symbol: NVDA
    shares: 5
    cost_basis: 500.0
    open_date: 2025-12-01
  - symbol: SPY
    shares: 20
    cost_basis: 520.0
    open_date: 2025-08-15
"""

_CSV = """symbol,shares,cost_basis,open_date
NVDA,10,450.5,2025-11-03
SPY,20,520.0,2025-08-15
"""


class TestYamlLoading:
    def test_roundtrip(self, tmp_path: Path):
        f = tmp_path / "p.local.yaml"
        f.write_text(_YAML, encoding="utf-8")
        pf = load_portfolio(f)
        assert pf.cash == 1000.5
        assert len(pf.positions) == 3
        assert pf.positions[0].symbol == "NVDA"  # 小写归一成大写
        assert pf.symbols == ["NVDA", "SPY"]  # 去重保序
        assert pf.total_cost() == pytest.approx(10 * 450.5 + 5 * 500.0 + 20 * 520.0)

    def test_empty_yaml_gives_empty_portfolio(self, tmp_path: Path):
        f = tmp_path / "p.yaml"
        f.write_text("", encoding="utf-8")
        pf = load_portfolio(f)
        assert pf.cash == 0.0 and pf.positions == []

    def test_top_level_list_rejected(self, tmp_path: Path):
        f = tmp_path / "p.yaml"
        f.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_portfolio(f)


class TestCsvLoading:
    def test_csv(self, tmp_path: Path):
        f = tmp_path / "p.csv"
        f.write_text(_CSV, encoding="utf-8")
        pf = load_portfolio(f)
        assert pf.cash == 0.0
        assert [p.symbol for p in pf.positions] == ["NVDA", "SPY"]

    def test_csv_with_bom_and_blank_lines(self, tmp_path: Path):
        f = tmp_path / "p.csv"
        f.write_text("\ufeff" + _CSV + "\n\n", encoding="utf-8")
        assert len(load_portfolio(f).positions) == 2


class TestValidation:
    def test_zero_shares_rejected(self):
        with pytest.raises(ValidationError, match="shares"):
            Position(symbol="A", shares=0, cost_basis=1.0, open_date="2025-01-02")

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            Position(symbol="A", shares=1, cost_basis=-1.0, open_date="2025-01-02")

    def test_blank_symbol_rejected(self):
        with pytest.raises(ValidationError):
            Position(symbol="  ", shares=1, cost_basis=1.0, open_date="2025-01-02")

    def test_unknown_field_rejected(self, tmp_path: Path):
        f = tmp_path / "p.yaml"
        f.write_text("positions:\n- symbol: A\n  shares: 1\n  cost_basis: 1\n  open_date: 2025-01-02\n  typo_field: 1\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_portfolio(f)

    def test_short_position_allowed_explicitly(self):
        p = Position(symbol="A", shares=-5, cost_basis=10.0, open_date="2025-01-02")
        assert p.shares == -5

    def test_missing_file_message_points_to_example(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="portfolio.example.yaml"):
            load_portfolio(tmp_path / "nope.yaml")

    def test_unsupported_extension(self, tmp_path: Path):
        f = tmp_path / "p.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="txt"):
            load_portfolio(f)


def test_example_file_in_repo_is_valid():
    """仓库自带的示例文件必须永远可加载（用户 copy 出发点）。"""
    example = Path(__file__).resolve().parents[2] / "portfolio.example.yaml"
    pf = load_portfolio(example)
    assert isinstance(pf, Portfolio)
    assert len(pf.positions) >= 1
