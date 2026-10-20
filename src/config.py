"""Đọc cấu hình dự án."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    """Đọc configs/config.yaml (hoặc file chỉ định) thành dict."""
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str) -> Path:
    """Trả về đường dẫn tuyệt đối của một thư mục khai báo trong cfg['paths']."""
    p = PROJECT_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
