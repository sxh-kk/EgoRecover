"""Resume the five processed validation assets using official ZIP byte ranges.

Run once at a time. Existing assets are checked against the archive CRC before
being reused. The source ZIP version and selected member list are pinned below.
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import threading
import time
import urllib.request
import zlib

URL = 'https://downloads.cs.stanford.edu/simurgh/chpatel/ee4d_motion_uniegomotion.zip'
ARCHIVE_SIZE = 35067467786
ROOT = Path(__file__).resolve().parents[1] / 'data'
STATE = ROOT / '.ee4d_validation_download'
CHUNKS = STATE / 'chunks'
SELECTED = {
    'ee4d_motion_uniegomotion/annotations/splits.json',
    'ee4d_motion_uniegomotion/takes.json',
    'ee4d_motion_uniegomotion/uniegomotion/ee_val.pt',
    'ee4d_motion_uniegomotion/uniegomotion/egoview_dinov2_val.pt',
    'ee4d_motion_uniegomotion/uniegomotion/v4_beta_ee_train_stats.pt',
}
BLOCK = 16 * 1024 * 1024
LOCK = threading.Lock()
LAST_ERROR = ''
RETRIES = 0


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def open_range(start, end):
    req = urllib.request.Request(URL, headers={
        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
        'User-Agent': 'EgoRecover-validation-data-download/1.0',
    })
    response = urllib.request.urlopen(req, timeout=45)
    expected = f'bytes {start}-{end}/{ARCHIVE_SIZE}'
    if response.status != 206 or response.headers.get('Content-Range') != expected:
        details = (response.status, response.headers.get('Content-Range'))
        response.close()
        raise RuntimeError(f'Unexpected partial response: {details}; expected {expected}')
    return response


def small_range(start, end):
    for attempt in range(10):
        try:
            with open_range(start, end) as r:
                data = r.read(end - start + 2)
            if len(data) != end - start + 1:
                raise IOError('Incomplete metadata range')
            return data
        except Exception as ex:
            print(f'{now()} metadata retry {attempt + 1}: {ex}', flush=True)
            if attempt == 9:
                raise
            time.sleep(min(2 ** attempt, 20))


def archive_manifest():
    start = ARCHIVE_SIZE - 65557
    tail = small_range(start, ARCHIVE_SIZE - 1)
    eocd = tail.rfind(b'PK\x05\x06')
    values = struct.unpack_from('<4s4H2LH', tail, eocd)
    count, size, offset = values[4:7]
    if count == 0xffff or size == 0xffffffff or offset == 0xffffffff:
        locator = tail.rfind(b'PK\x06\x07', 0, eocd)
        zo = struct.unpack_from('<4sLQL', tail, locator)[2]
        z = tail[zo-start:zo-start+56] if zo >= start else small_range(zo, zo+55)
        v = struct.unpack('<4sQ2H2L4Q', z)
        count, size, offset = v[7:10]
    central = tail[offset-start:offset-start+size] if offset >= start else small_range(offset, offset+size-1)
    pos, members = 0, []
    while pos + 46 <= len(central) and central[pos:pos+4] == b'PK\x01\x02':
        v = struct.unpack_from('<4s6H3L5H2L', central, pos)
        compressed, unpacked, fl, xl, cl, local = v[8], v[9], v[10], v[11], v[12], v[16]
        name = central[pos+46:pos+46+fl].decode('utf-8')
        extra = central[pos+46+fl:pos+46+fl+xl]
        p = 0
        while p + 4 <= len(extra):
            tag, length = struct.unpack_from('<HH', extra, p)
            data = extra[p+4:p+4+length]
            if tag == 1:
                q = 0
                if unpacked == 0xffffffff:
                    unpacked = struct.unpack_from('<Q', data, q)[0]
                    q += 8
                if compressed == 0xffffffff:
                    compressed = struct.unpack_from('<Q', data, q)[0]
                    q += 8
                if local == 0xffffffff:
                    local = struct.unpack_from('<Q', data, q)[0]
            p += 4 + length
        if name in SELECTED:
            header = small_range(local, local+29)
            h = struct.unpack('<4s5H3L2H', header)
            if h[0] != b'PK\x03\x04' or v[4] != 8:
                raise RuntimeError(f'Unsupported ZIP member: {name}')
            members.append(dict(name=name, compressed_bytes=compressed, uncompressed_bytes=unpacked,
                                crc32=v[7], data_offset=local+30+h[9]+h[10]))
        pos += 46 + fl + xl + cl
    if {m['name'] for m in members} != SELECTED:
        raise RuntimeError('Expected archive members are missing')
    return members


def chunk_path(member, index):
    return CHUNKS / (Path(member['name']).name + f'.{index:05d}')


def fetch_chunk(member, index):
    global RETRIES, LAST_ERROR
    begin = index * BLOCK
    length = min(BLOCK, member['compressed_bytes'] - begin)
    path = chunk_path(member, index)
    done = path.stat().st_size if path.exists() else 0
    if done > length:
        raise RuntimeError(f'Oversized cached chunk {path}')
    if done == length:
        return
    failures = 0
    while done < length:
        try:
            first = member['data_offset'] + begin + done
            last = member['data_offset'] + begin + length - 1
            with open_range(first, last) as r, path.open('ab') as f:
                while done < length:
                    block = r.read(min(1024*1024, length-done))
                    if not block:
                        raise IOError('Connection ended before requested bytes arrived')
                    f.write(block)
                    f.flush()
                    done += len(block)
            failures = 0
        except Exception as ex:
            done = path.stat().st_size if path.exists() else 0
            failures += 1
            with LOCK:
                RETRIES += 1
                LAST_ERROR = f'{Path(member["name"]).name} chunk {index}: {type(ex).__name__}: {ex}'
            if failures >= 20:
                raise RuntimeError(LAST_ERROR) from ex
            time.sleep(min(2 ** min(failures, 5), 30) + random.random())


def unpack(member):
    target = ROOT / member['name']
    if target.exists():
        return verify_existing(member)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.download')
    decoder = zlib.decompressobj(-15)
    crc, length = 0, 0
    sha = hashlib.sha256()
    with temporary.open('wb') as output:
        for index in range((member['compressed_bytes'] + BLOCK - 1) // BLOCK):
            with chunk_path(member, index).open('rb') as source:
                while True:
                    compressed = source.read(1024*1024)
                    if not compressed:
                        break
                    while compressed:
                        plain = decoder.decompress(compressed, 8*1024*1024)
                        compressed = decoder.unconsumed_tail
                        output.write(plain)
                        crc = zlib.crc32(plain, crc)
                        sha.update(plain)
                        length += len(plain)
        plain = decoder.flush()
        output.write(plain)
        crc = zlib.crc32(plain, crc)
        sha.update(plain)
        length += len(plain)
    if not decoder.eof or decoder.unused_data or length != member['uncompressed_bytes'] or crc != member['crc32']:
        raise RuntimeError(f'ZIP integrity check failed for {member["name"]}: length={length}, crc={crc}')
    temporary.replace(target)
    return dict(path=str(target), bytes=length, crc32=f'{crc:08x}', sha256=sha.hexdigest())


def verify_existing(member):
    target = ROOT / member['name']
    sha, crc, length = hashlib.sha256(), 0, 0
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            length += len(block)
    if length != member['uncompressed_bytes'] or crc != member['crc32']:
        raise ValueError(f'Existing asset failed archive CRC/length; no overwrite performed: {target}')
    return dict(path=str(target), bytes=length, crc32=f'{crc:08x}', sha256=sha.hexdigest())


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    CHUNKS.mkdir(exist_ok=True)
    manifest_path = STATE / 'archive_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source'] != URL or manifest['archive_bytes'] != ARCHIVE_SIZE:
            raise ValueError('Cached manifest belongs to another archive version')
        members = manifest['members']
        if {m['name'] for m in members} != SELECTED:
            raise ValueError('Unexpected cached member list')
    else:
        print(f'{now()} Reading official ZIP directory', flush=True)
        members = archive_manifest()
        atomic_json(manifest_path, dict(source=URL, archive_bytes=ARCHIVE_SIZE, selected_at=now(), members=members))
    # A interrupted extraction may already have published some verified files.
    existing = [verify_existing(m) for m in members if (ROOT / m['name']).exists()]
    remaining = [m for m in members if not (ROOT / m['name']).exists()]
    total = sum(m['compressed_bytes'] for m in members)
    if not remaining:
        status_path = STATE / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update(phase='complete', updated_at=now(), completed_at=status.get('completed_at', now()),
                      source=URL, verified_files=existing, downloaded_bytes=total,
                      compressed_bytes_total=total, progress=1, eta_seconds=0)
        atomic_json(status_path, status)
        print(f'{now()} COMPLETE: all {len(existing)} existing assets verified; no download needed', flush=True)
        return
    status = dict(phase='downloading', started_at=now(), source=URL, compressed_bytes_total=total,
                  uncompressed_bytes_total=sum(m['uncompressed_bytes'] for m in members), files=len(members), workers=3)
    atomic_json(STATE / 'status.json', status)
    print(f'{now()} Downloading {len(members)} selected members: {total/1e9:.3f} GB compressed', flush=True)
    started = time.monotonic()
    completed_bytes = sum(m['compressed_bytes'] for m in members if (ROOT / m['name']).exists())
    jobs = [(m, i) for m in sorted(remaining, key=lambda m:m['compressed_bytes'])
            for i in range((m['compressed_bytes'] + BLOCK - 1)//BLOCK)]
    def downloaded_bytes():
        return completed_bytes + sum(chunk_path(m, i).stat().st_size for m, i in jobs if chunk_path(m, i).exists())
    initial_bytes = downloaded_bytes()
    samples = [(started, initial_bytes)]
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fetch_chunk, m, i) for m, i in jobs}
        last_print = 0
        while pending:
            done, pending = cf.wait(pending, timeout=5, return_when=cf.FIRST_EXCEPTION)
            for future in done:
                future.result()
            count = downloaded_bytes()
            tick = time.monotonic()
            samples.append((tick, count))
            while len(samples) > 2 and tick - samples[0][0] > 120:
                samples.pop(0)
            rate = (count-samples[0][1]) / max(tick-samples[0][0], 1)
            eta = (total-count)/rate if rate>0 else None
            status.update(updated_at=now(), downloaded_bytes=count, progress=count/total,
                          elapsed_seconds=round(tick-started,1), recent_bytes_per_second=rate,
                          eta_seconds=eta, eta_at=(dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))+dt.timedelta(seconds=eta)).isoformat(timespec='seconds') if eta is not None else None,
                          retries=RETRIES, last_error=LAST_ERROR)
            atomic_json(STATE / 'status.json', status)
            if tick-last_print >= 30 or not pending:
                print(f'{now()} {count/1e9:.3f}/{total/1e9:.3f} GB ({count/total:.1%}), {rate/1048576:.2f} MiB/s, ETA {status["eta_at"]}, retries {RETRIES}', flush=True)
                last_print = tick
    verified = existing.copy()
    status['phase'] = 'extracting_and_verifying'
    atomic_json(STATE / 'status.json', status)
    for member in sorted(remaining, key=lambda m:m['compressed_bytes']):
        print(f'{now()} Extracting and verifying {member["name"]}', flush=True)
        verified.append(unpack(member))
        status.update(verified_files=verified, updated_at=now())
        atomic_json(STATE / 'status.json', status)
    # Delete only the temporary compressed chunks created by this downloader, after every member verifies.
    for member in members:
        for index in range((member['compressed_bytes'] + BLOCK - 1)//BLOCK):
            chunk_path(member, index).unlink(missing_ok=True)
    status.update(phase='complete', completed_at=now(), updated_at=now(), verified_files=verified,
                  eta_seconds=0, temporary_chunks_removed=True)
    atomic_json(STATE / 'status.json', status)
    print(f'{now()} COMPLETE: {len(verified)} files verified by ZIP CRC32; SHA256 recorded in {STATE / "status.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.data_dir.resolve()
    STATE, CHUNKS = ROOT / '.ee4d_validation_download', ROOT / '.ee4d_validation_download' / 'chunks'
    STATE.mkdir(parents=True, exist_ok=True)
    worker_lock = (STATE / 'worker.lock').open('a')
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another validation downloader is active for this data directory')
    try:
        main()
    except Exception as ex:
        STATE.mkdir(parents=True, exist_ok=True)
        path = STATE / 'status.json'
        status = json.loads(path.read_text()) if path.exists() else {}
        status.update(phase='error', updated_at=now(), error=f'{type(ex).__name__}: {ex}')
        atomic_json(path, status)
        print(f'{now()} ERROR: {type(ex).__name__}: {ex}', flush=True)
        raise
