import hashlib
import io
import zlib

import pytest

from data_pipeline import download_validation as download


def test_zip_extract_crc_and_existing_file_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ROOT", tmp_path)
    monkeypatch.setattr(download, "CHUNKS", tmp_path / "chunks")
    monkeypatch.setattr(download, "BLOCK", 19)
    download.CHUNKS.mkdir()
    original = bytes(range(256))*50
    encoder = zlib.compressobj(wbits=-15)
    compressed = encoder.compress(original)+encoder.flush()
    member = dict(name="test/payload.bin", compressed_bytes=len(compressed),
                  uncompressed_bytes=len(original), crc32=zlib.crc32(original), data_offset=42)
    for index, offset in enumerate(range(0, len(compressed), download.BLOCK)):
        download.chunk_path(member, index).write_bytes(compressed[offset:offset+download.BLOCK])
    result = download.unpack(member)
    assert result["sha256"] == hashlib.sha256(original).hexdigest()
    assert download.unpack(member) == result
    (tmp_path / member["name"]).write_bytes(original[:-1])
    with pytest.raises(ValueError, match="CRC/length"):
        download.unpack(member)


def test_download_resumes_exact_partial_byte_range(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "CHUNKS", tmp_path)
    monkeypatch.setattr(download, "BLOCK", 8)
    member = dict(name="payload.bin", compressed_bytes=16, data_offset=100)
    download.chunk_path(member, 1).write_bytes(b"ab")
    calls = []

    def open_range(first, last):
        calls.append((first, last))
        return io.BytesIO(b"cdefgh")

    monkeypatch.setattr(download, "open_range", open_range)
    download.fetch_chunk(member, 1)
    assert calls == [(110, 115)]
    assert download.chunk_path(member, 1).read_bytes() == b"abcdefgh"
    download.fetch_chunk(member, 1)
    assert len(calls) == 1
