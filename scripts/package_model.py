"""Build an allowlisted model bundle for Hugging Face or GitHub Releases."""
import argparse
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("model.safetensors", "rl_agent_config.json", "best-epoch.json",
            "training-metrics.json", "encoder/config.json",
            "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != "limit_remaining"}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"(?:/Users/|/home/)[^\s\"']+", lambda m: Path(m[0]).name, value)
    return value


def package(source, destination, archive=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    for name in REQUIRED:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    destination.mkdir(parents=True, exist_ok=False)
    names = list(REQUIRED)
    if (source / "provenance.json").exists():
        names.append("provenance.json")
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".json"):
            target.write_text(json.dumps(clean(json.loads((source / name).read_text())), indent=2) + "\n")
        else:
            shutil.copyfile(source / name, target)
    for src, dest in (("LICENSE", "LICENSE"), ("NOTICE", "NOTICE"),
                      ("model-card/README.md", "README.md")):
        shutil.copyfile(ROOT / src, destination / dest)
    manifest = {p.relative_to(destination).as_posix(): digest(p)
                for p in sorted(destination.rglob("*")) if p.is_file()}
    (destination / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in manifest.items()))
    if archive:
        path = destination.with_suffix(".tar")
        if path.exists():
            raise FileExistsError(path)
        with tarfile.open(path, "w") as tar:
            for file in sorted(destination.rglob("*")):
                if file.is_file():
                    info = tar.gettarinfo(str(file), arcname=f"{destination.name}/{file.relative_to(destination)}")
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with file.open("rb") as stream:
                        tar.addfile(info, stream)
        archive_hash = digest(path)
        path.with_suffix(".tar.sha256").write_text(f"{archive_hash}  {path.name}\n")
        print(f"Archive: {path} ({path.stat().st_size:,} bytes)")
    print(f"Bundle: {destination} ({len(manifest)} files + SHA256SUMS)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/laya-doom-v3")
    p.add_argument("--output", type=Path, default=ROOT / "dist/laya-doom-v3")
    p.add_argument("--archive", action="store_true")
    a = p.parse_args()
    package(a.checkpoint, a.output, a.archive)
