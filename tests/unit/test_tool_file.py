import pytest

import connic
from connic import ToolFile


def test_tool_file_is_exported_from_public_package():
    assert connic.ToolFile is ToolFile
    assert "ToolFile" in connic.__all__


def test_tool_file_constructor_is_keyword_only():
    with pytest.raises(TypeError):
        ToolFile("text/plain", data=b"file")


def test_tool_file_infers_inline_data_size():
    file = ToolFile(
        mime_type="application/pdf",
        name="invoice.pdf",
        data=b"pdf-data",
    )

    assert file.data == b"pdf-data"
    assert file.size_bytes == 8
    assert file.uri is None


def test_tool_file_supports_uri_source():
    file = ToolFile(
        mime_type="application/pdf",
        name="invoice.pdf",
        size_bytes=42,
        uri="https://files.example.com/invoice.pdf",
    )

    assert file.uri == "https://files.example.com/invoice.pdf"
    assert file.size_bytes == 42


@pytest.mark.parametrize(
    "sources",
    [
        {},
        {"data": b"file", "uri": "https://files.example.com/file"},
    ],
)
def test_tool_file_requires_exactly_one_source(sources):
    with pytest.raises(ValueError, match="exactly one of data or uri"):
        ToolFile(mime_type="application/octet-stream", **sources)


def test_tool_file_rejects_inline_size_mismatch():
    with pytest.raises(ValueError, match="size_bytes must match the data length"):
        ToolFile(
            mime_type="application/octet-stream",
            data=b"file",
            size_bytes=3,
        )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"mime_type": "", "data": b"file"}, ValueError, "mime_type"),
        (
            {"mime_type": "text/plain", "name": "", "data": b"file"},
            ValueError,
            "name",
        ),
        (
            {"mime_type": "text/plain", "size_bytes": -1, "uri": "https://example.com/file"},
            ValueError,
            "size_bytes",
        ),
        (
            {"mime_type": "text/plain", "data": "not-bytes"},
            TypeError,
            "data must be bytes",
        ),
        ({"mime_type": "text/plain", "uri": ""}, ValueError, "uri"),
    ],
)
def test_tool_file_rejects_invalid_metadata_and_sources(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        ToolFile(**kwargs)
