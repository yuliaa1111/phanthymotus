"""Atomic, checksum-verified Serena release download from the model file server."""
import fcntl
import hashlib
import json
import platform
import sys
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release(root):
    root = Path(root)
    receipt = json.loads((root / 'release.json').read_text())
    if receipt.get('format_version') != 1 or not receipt.get('files'):
        raise ValueError('Invalid Serena release receipt')
    for name, expected in receipt['files'].items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid release path: ' + name)
        target = root / name
        if target.is_symlink() or target.stat().st_size != expected['bytes'] or sha256(target) != expected['sha256']:
            raise ValueError('Serena release checksum mismatch: ' + name)
    return receipt


def check_platform(expected):
    if expected.get('architecture') != platform.machine():
        raise RuntimeError('Serena release requires ' + str(expected.get('architecture')))
    python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    if expected.get('python') != python_version:
        raise RuntimeError('Serena release requires Python ' + str(expected.get('python')))
    import tensorrt
    if expected.get('tensorrt') != str(tensorrt.__version__):
        raise RuntimeError('Serena release requires TensorRT ' + str(expected.get('tensorrt')))


def ensure_release(model_dir, manifest_path, on_status=None):
    root = Path(model_dir)
    # Explicitly mounted developer models remain supported. Published bundles
    # carry release.json and get complete file verification at process startup.
    if (root / 'config.json').exists():
        if (root / 'release.json').exists():
            receipt = verify_release(root)
            check_platform(receipt['platform'])
        return root
    manifest = json.loads(Path(manifest_path).read_text())
    check_platform(manifest['platform'])
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + '.download.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / 'config.json').exists():
            verify_release(root)
            return root
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Refusing to replace nonempty model directory: ' + str(root))
        if on_status:
            on_status('Downloading Serena JP6 model and frontend assets')
        with tempfile.TemporaryDirectory(prefix='.serena-', dir=root.parent) as tmp:
            temp = Path(tmp)
            archive_path = temp / 'release.tar.gz'
            with urllib.request.urlopen(manifest['url'], timeout=60) as response, archive_path.open('xb') as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if archive_path.stat().st_size != manifest['bytes'] or sha256(archive_path) != manifest['sha256']:
                raise ValueError('Serena archive checksum mismatch')
            staging = temp / 'unpacked'
            staging.mkdir()
            with tarfile.open(archive_path, 'r:gz') as archive:
                members = archive.getmembers()
                for item in members:
                    path = PurePosixPath(item.name)
                    if path.is_absolute() or '..' in path.parts or not (item.isfile() or item.isdir()):
                        raise ValueError('Invalid Serena archive member: ' + item.name)
                archive.extractall(staging, members=members)
            receipt = verify_release(staging)
            check_platform(receipt['platform'])
            if receipt['candidate'] != manifest['candidate']:
                raise ValueError('Serena candidate identity mismatch')
            if root.exists():
                root.rmdir()
            staging.rename(root)
        if on_status:
            on_status('Serena release checksum verified')
    return root
