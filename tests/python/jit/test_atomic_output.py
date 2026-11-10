# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""CPU-only tests for mori.jit.core.atomic_output (no GPU, no compiled extension)."""

import threading
import time

import pytest

from mori.jit.core import atomic_output

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


def _write_chunked(path, data, chunk=4096, delay=0.0):
    with open(path, "wb") as f:
        for i in range(0, len(data), chunk):
            f.write(data[i : i + chunk])
            f.flush()
            if delay:
                time.sleep(delay)


def test_target_appears_only_after_exit(tmp_path):
    target = tmp_path / "k.hsaco"
    with atomic_output(target) as tmp:
        _write_chunked(tmp, PAYLOAD)
        assert not target.exists()
    assert target.read_bytes() == PAYLOAD
    assert list(tmp_path.iterdir()) == [target]


def test_exception_leaves_no_target_and_no_temp(tmp_path):
    target = tmp_path / "k.hsaco"
    with pytest.raises(RuntimeError):
        with atomic_output(target) as tmp:
            tmp.write_bytes(PAYLOAD[:100])
            raise RuntimeError("compiler failed")
    assert list(tmp_path.iterdir()) == []


def test_temp_is_unique_hidden_sibling_not_matching_suffix(tmp_path):
    target = tmp_path / "k.hsaco"
    with atomic_output(target) as a, atomic_output(target) as b:
        a.touch()
        b.touch()
    assert a != b
    for t in (a, b):
        assert t.parent == target.parent
        assert t.name.startswith(".k.hsaco.")
        assert t.suffix != target.suffix


def test_existing_target_replaced(tmp_path):
    target = tmp_path / "lib.bc"
    target.write_bytes(b"old")
    with atomic_output(target) as tmp:
        tmp.write_bytes(PAYLOAD[:10])
        assert target.read_bytes() == b"old"
    assert target.read_bytes() == PAYLOAD[:10]


def test_concurrent_reader_never_sees_partial_file(tmp_path):
    target = tmp_path / "k.hsaco"
    done = threading.Event()
    seen = []

    def reader():
        while True:
            finished = done.is_set()
            if target.is_file():
                seen.append(target.read_bytes())
            if finished:
                return

    def writer():
        with atomic_output(target) as tmp:
            _write_chunked(tmp, PAYLOAD, chunk=64 * 1024, delay=0.005)
        done.set()

    threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert seen and all(s == PAYLOAD for s in seen)
