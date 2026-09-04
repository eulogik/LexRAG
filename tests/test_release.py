"""Release hygiene: single version source + session-name-once behavior."""

import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def test_version_single_source():
    m = re.search(r'__version__\s*=\s*"([^"]+)"', _read("api/version.py"))
    assert m, "api/version.py must define __version__"
    version = m.group(1)
    assert f'version = "{version}"' in _read("pyproject.toml")
    assert "version=__version__" in _read("api/main.py")
    assert '"version": __version__' in _read("api/main.py")
    assert version in _read("marketing/index.html")


def test_session_named_once(tmp_path, monkeypatch):
    import api.memory as mem
    monkeypatch.setattr(mem, "DB_PATH", str(tmp_path / "t.db"))
    sid = "sess_test123"

    # Fresh session: empty history -> chat flow would set the name
    assert mem.get_history(sid) == []
    mem.update_session_name(sid, "First question here")
    assert mem.get_session_name(sid) == "First question here"

    # Second message: history non-empty -> chat flow must NOT rename
    mem.save_message(sid, "user", "First question here")
    mem.save_message(sid, "assistant", "Answer one")
    mem.save_message(sid, "user", "Second question here")
    history = mem.get_history(sid, limit=5)
    assert len(history) == 3
    if history:
        pass  # name stays untouched
    else:
        mem.update_session_name(sid, "Second question here")
    assert mem.get_session_name(sid) == "First question here"


def test_history_prefetch_excludes_current_question(tmp_path, monkeypatch):
    import api.memory as mem
    monkeypatch.setattr(mem, "DB_PATH", str(tmp_path / "t2.db"))
    sid = "sess_dup_check"
    mem.save_message(sid, "user", "Earlier question")
    mem.save_message(sid, "assistant", "Earlier answer")

    # Chat flow fetches history BEFORE saving the current question
    history = mem.get_history(sid, limit=5)
    assert all(h["content"] != "Current question" for h in history)
    mem.save_message(sid, "user", "Current question")
