#!/usr/bin/env python
"""One file, one command: set up and submit the two SPICE isolation runs.

    scp setup_spice_iso.py  <cluster>:/ssd/work/linruifei/DPA/
    ssh <cluster> "cd /ssd/work/linruifei/DPA && python setup_spice_iso.py"

That is the whole procedure.  This script creates both run directories,
copies the model and trainer out of the existing v10e run, writes the
configs and the submit scripts itself, and submits both jobs.

WHAT THE TWO RUNS ARE FOR
-------------------------
v10e on SPICE has produced NaN twice, at steps 11000 and 5000.  The
eager NaN hunt then passed 17000 steps without failing -- but it had
changed TWO things at once (use_compile off AND batch halved), so its
survival does not say which one mattered.

  isoA  compile ON, amp ON, batch halved
        Only the batch differs from the crashing config.  If this still
        dies, the batch is exonerated and compilation is what differs
        from the eager hunt.

  isoB  compile ON, amp OFF, batch halved
        Identical to isoA except bf16 is gone.  The reservoir runs 50+
        sequential matmuls wrapped in
        torch.autocast(device_type="cuda", enabled=False).  That guard
        works in eager, but it is not established that it survives
        make_fx(tracing_mode="symbolic"), which is how SeZM compiles.
        If it is traced away, the statevector evolves in 8-mantissa-bit
        bf16 and the gradients through it are hopeless.  isoA dying
        while isoB lives would pin that down.

Either surviving run is also a usable production run: isoA is the one
that deviates least from the benchmark config (batch only).

  --no-submit   set everything up but do not sbatch
  --steps N     override numb_steps
  --batch SPEC  override the halved batch (default auto:1000)
"""

import json
import os
import shutil
import subprocess
import sys

SUBMIT = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --partition=8-5090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output={job}_%j.log

# {why}

cd {rundir}

export DP_ENABLE_TF=0
export LD_LIBRARY_PATH=/data/home/linruifei/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/data/home/linruifei/.conda/envs/dpa4/lib:$LD_LIBRARY_PATH
export TORCH_LINALG_PREFER_MAGMA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "QRC-DPA4 v10e  |  SPICE  |  isolation run {tag}"
echo "  use_compile = {compile}   use_amp = {amp}   batch = {batch}"
echo "run dir : $(pwd)"
echo "config  : {cfg}"
echo "model   : $(md5sum qrc_dpa4_model.py)"
echo "trainer : $(md5sum train_qrc_dpa4.py)"
echo "============================================"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
echo "============================================"

python train_qrc_dpa4.py --pt train {cfg} 2>&1
"""

VARIANTS = [
    ("A", True, True,
     "isoA: only the batch differs from the config that produced NaN. "
     "If this still dies,\n# the batch is exonerated and compilation is "
     "the remaining difference."),
    ("B", True, False,
     "isoB: same as isoA but with bf16 off.  The reservoir's 50+ "
     "sequential matmuls are\n# guarded by "
     "torch.autocast(enabled=False), which works in eager but may not "
     "survive\n# make_fx symbolic tracing.  isoA dying while isoB lives "
     "pins that down."),
]


def main():
    argv = sys.argv[1:]
    submit = "--no-submit" not in argv
    if "--no-submit" in argv:
        argv.remove("--no-submit")

    steps = None
    if "--steps" in argv:
        i = argv.index("--steps")
        steps = int(argv[i + 1])
        del argv[i:i + 2]

    batch = "auto:1000"
    if "--batch" in argv:
        i = argv.index("--batch")
        batch = argv[i + 1]
        del argv[i:i + 2]

    base = os.path.abspath(argv[0]) if argv else os.getcwd()
    src_dir = os.path.join(base, "qrc_dpa4_v10e_run")

    need = ["qrc_dpa4_model.py", "train_qrc_dpa4.py",
            "input_qrc_dpa4_equi.json"]

    missing = [f for f in need
               if not os.path.isfile(os.path.join(src_dir, f))]

    if missing:
        print("cannot find {} in {}".format(missing, src_dir))
        print("pass the parent directory as the first argument if the "
              "layout differs")
        return 1

    with open(os.path.join(src_dir, "input_qrc_dpa4_equi.json")) as fh:
        base_cfg = json.load(fh)

    print("=" * 70)
    print("  source run dir : {}".format(src_dir))
    print("  base config    : use_compile={} use_amp={} batch={}".format(
        base_cfg["model"].get("use_compile"),
        base_cfg["model"]["descriptor"].get("use_amp"),
        base_cfg["training"]["training_data"].get("batch_size")))
    print("=" * 70)

    jobs = []

    for tag, use_compile, use_amp, why in VARIANTS:
        run = "qrc_dpa4_v10e_iso{}_run".format(tag)
        rundir = os.path.join(base, run)
        os.makedirs(rundir, exist_ok=True)

        for f in ("qrc_dpa4_model.py", "train_qrc_dpa4.py"):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(rundir, f))

        cfg = json.loads(json.dumps(base_cfg))
        cfg["_comment"] = "QRC-DPA4 v10e SPICE -- NaN isolation run " + tag
        cfg["model"]["use_compile"] = use_compile
        cfg["model"]["descriptor"]["use_amp"] = use_amp
        cfg["training"]["training_data"]["batch_size"] = batch
        cfg["training"]["save_freq"] = 2000
        if steps is not None:
            cfg["training"]["numb_steps"] = steps

        cfg_name = "input_qrc_dpa4_equi_iso{}.json".format(tag)
        with open(os.path.join(rundir, cfg_name), "w") as fh:
            json.dump(cfg, fh, indent=2)

        job = "qrc_v10e_iso{}".format(tag)
        sh_name = "submit_{}.sh".format(job)
        sh = SUBMIT.format(job=job, why=why, rundir=rundir, tag=tag,
                           cfg=cfg_name, compile=use_compile, amp=use_amp,
                           batch=batch)

        if not all(ord(c) < 128 for c in sh):
            print("refusing to write non-ASCII submit script")
            return 1

        sh_path = os.path.join(rundir, sh_name)
        with open(sh_path, "w") as fh:
            fh.write(sh)
        os.chmod(sh_path, 0o755)

        diff = []
        if use_compile != base_cfg["model"].get("use_compile"):
            diff.append("use_compile")
        if use_amp != base_cfg["model"]["descriptor"].get("use_amp"):
            diff.append("use_amp")
        if batch != base_cfg["training"]["training_data"].get("batch_size"):
            diff.append("batch_size")

        print("  iso{}  {}".format(tag, rundir))
        print("        use_compile={}  use_amp={}  batch={}".format(
            use_compile, use_amp, batch))
        print("        differs from base in: {}".format(diff))

        jobs.append((tag, rundir, sh_name))

    print("=" * 70)

    if not submit:
        print("  --no-submit given; submit by hand with:")
        for tag, rundir, sh in jobs:
            print("    (cd {} && sbatch {})".format(rundir, sh))
        return 0

    for tag, rundir, sh in jobs:
        try:
            out = subprocess.run(
                ["sbatch", sh], cwd=rundir,
                capture_output=True, text=True, check=False
            )
            line = (out.stdout or out.stderr).strip()
            print("  submitted iso{}: {}".format(tag, line))
        except FileNotFoundError:
            print("  sbatch not found; run by hand:")
            print("    (cd {} && sbatch {})".format(rundir, sh))

    print("=" * 70)
    print("  watch both with:")
    print("    tail -f {}/qrc_v10e_isoA_*.log".format(
        os.path.join(base, "qrc_dpa4_v10e_isoA_run")))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
