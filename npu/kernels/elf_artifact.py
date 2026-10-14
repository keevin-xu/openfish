"""Shared helper: compile an air module to a full ELF + <name>.json manifest for openfish nn_npu.cpp."""
import hashlib
import json
import re
import shutil
from pathlib import Path


def compile_elf(module, name, out_dir, fields):
    from air.backend.xrt import XRTBackend

    funcs = [f for f in re.findall(r"func\.func @(\w+)\(", str(module))]
    backend = XRTBackend(target_device="npu2", omit_while_true_loop=False, output_format="elf",
                         instance_name=funcs[0])
    artifact = backend.compile(module)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    elf = f"{name}.elf"
    shutil.copyfile(artifact.output_binary, out / elf)
    manifest = {"kernel": name, "file": elf, "format": "elf", "kernel_name": artifact.kernel, "dtype": "bf16",
                **fields, "sha256": hashlib.sha256((out / elf).read_bytes()).hexdigest()}
    (out / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))
