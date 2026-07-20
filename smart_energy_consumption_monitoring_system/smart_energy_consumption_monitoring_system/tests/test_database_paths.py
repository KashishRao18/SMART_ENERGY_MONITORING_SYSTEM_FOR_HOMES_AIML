import os
from pathlib import Path

import pytest

from app.utils import database


def test_get_user_data_dir_creates_user_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(database, "USER_DATA_DIR", str(tmp_path))
    path = database.get_user_data_dir("user_123")
    assert path == os.path.join(str(tmp_path), "user_123")
    assert os.path.isdir(path)


def test_get_user_data_dir_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(database, "USER_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        database.get_user_data_dir("../etc/passwd")

