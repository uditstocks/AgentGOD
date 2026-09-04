"""Tests for reading the files a task names.

Generated agents cannot open files and should not be able to. The session
can, because the user naming a file at their own prompt is not untrusted code
reaching for the disk - but only files they actually named, and never a
secret.
"""

from __future__ import annotations

import attachments
from attachments import MAX_FILES, attach, find_references


def write(root, name, text="hello world"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- finding what the user pointed at ------------------------------------------


def test_an_at_reference_is_always_taken(tmp_path):
    write(tmp_path, "notes.md")
    assert "notes.md" in find_references("summarise @notes.md", tmp_path)


def test_a_bare_filename_is_taken_when_the_file_exists(tmp_path):
    write(tmp_path, "README.md")
    assert "README.md" in find_references("read README.md and tell me the name", tmp_path)


def test_a_bare_filename_that_does_not_exist_is_only_a_mention(tmp_path):
    """"the fix landed in main.py" is not a request to read main.py."""
    assert find_references("the fix landed in main.py", tmp_path) == []


def test_a_quoted_path_is_taken(tmp_path):
    write(tmp_path, "report.txt")
    assert "report.txt" in find_references('summarise "report.txt"', tmp_path)


def test_a_task_with_no_files_finds_nothing(tmp_path):
    assert find_references("write a haiku about rain", tmp_path) == []


# --- reading ------------------------------------------------------------------


def test_the_contents_are_put_in_front_of_the_task(tmp_path):
    write(tmp_path, "brief.md", "The project is called Kestrel.")

    result = attach("read brief.md and tell me the project name", tmp_path)

    assert result.any_read
    assert "The project is called Kestrel." in result.task
    assert "read brief.md and tell me the project name" in result.task


def test_a_task_with_no_readable_file_is_unchanged(tmp_path):
    task = "summarise @missing.md"
    result = attach(task, tmp_path)

    assert not result.any_read
    assert result.task == task
    assert result.files[0].problem == "not found"


def test_nothing_referenced_means_nothing_touched(tmp_path):
    task = "write a haiku about rain"
    assert attach(task, tmp_path).task == task


def test_secrets_are_refused_even_when_named(tmp_path):
    """An API key must not be uploaded to a model provider by accident."""
    write(tmp_path, ".env", "ANTHROPIC_API_KEY=sk-ant-secret")

    result = attach("what is in @.env", tmp_path)

    assert not result.any_read
    assert "sk-ant-secret" not in result.task
    assert "secret" in result.files[0].problem


def test_a_binary_file_is_not_read(tmp_path):
    (tmp_path / "logo.svg").write_bytes(b"\x00\x01\x02binary")
    result = attach("describe @logo.svg", tmp_path)
    assert not result.any_read
    assert "binary" in result.files[0].problem


def test_a_large_file_is_truncated_rather_than_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments, "MAX_BYTES_PER_FILE", 100)
    write(tmp_path, "big.txt", "x" * 5000)

    result = attach("summarise @big.txt", tmp_path)

    assert result.any_read
    assert result.files[0].truncated
    assert "[...truncated...]" in result.task


def test_no_more_than_the_file_limit_is_read(tmp_path):
    names = [f"file{index}.txt" for index in range(MAX_FILES + 3)]
    for name in names:
        write(tmp_path, name)

    result = attach(" ".join(f"@{name}" for name in names), tmp_path)

    assert len(result.read) <= MAX_FILES


def test_every_file_read_is_labelled_for_the_user(tmp_path):
    """The contents are about to leave the machine; the user is told which."""
    write(tmp_path, "notes.md", "hello")
    result = attach("summarise @notes.md", tmp_path)
    assert "notes.md" in result.files[0].label()
