"""Serve the pinned original Laya or a separate Doom-adapted checkpoint."""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
from laya_runtime import BASE_REPO, BASE_REVISION, LAYA_SOURCE_COMMIT, base_checkpoint, choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Local adapted checkpoint directory")
    parser.add_argument("--model", choices=["typed-decisions", "doom-adapted"], default="doom-adapted")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.model == "doom-adapted" and not args.checkpoint:
        parser.error("--checkpoint is required for doom-adapted")
    if args.model == "typed-decisions" and args.checkpoint:
        parser.error("--checkpoint is reserved for doom-adapted; original weights use the pinned revision")
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from laya import Agent

    torch.set_num_threads(4)
    device = choose_device(args.device)
    root = base_checkpoint(args.checkpoint)
    if args.model == "doom-adapted" and not (root / "best-epoch.json").exists():
        raise RuntimeError("No epoch improved held-out validation")
    model = Agent(str(root), device=device)
    lock = threading.Lock()
    app = FastAPI()
    with (root / "model.safetensors").open("rb") as handle:
        weights_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {"status": "ok", "models": {args.model: device},
                "checkpoint": root.name, "weights_sha256": weights_hash,
                "laya_source_commit": LAYA_SOURCE_COMMIT}
    if args.model == "typed-decisions":
        metadata.update(repo=BASE_REPO, revision=BASE_REVISION)
    elif (root / "training-metrics.json").exists():
        metadata["training"] = json.loads((root / "training-metrics.json").read_text())

    class Request(BaseModel):
        state: str
        questions: dict
        model: str = args.model

    @app.get("/health")
    def health():
        return metadata

    @app.post("/predict")
    def predict(request: Request):
        if request.model != args.model:
            raise HTTPException(400, "Requested model is not loaded by this server")
        started = time.perf_counter()
        with lock:
            result = model.predict(request.state, request.questions)
        result["routing"] = {"model": args.model, "checkpoint": root.name,
                             "weights_sha256": weights_hash}
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    port = args.port if args.port is not None else (8001 if args.model == "doom-adapted" else 8000)
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
