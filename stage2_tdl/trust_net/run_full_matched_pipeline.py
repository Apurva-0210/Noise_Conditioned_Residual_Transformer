"""
run_full_matched_pipeline.py
────────────────────────────────────────────────────────────────────────
End-to-end pipeline:
  1. Train TRUST-Net V2 to exact NCRN parity (30 epochs, 6000 samples).
  2. Run Head-to-Head Verification (Full-Pilot & Sparse-Pilot).
  3. Generate Residual Noise Heatmaps and Error Maps.
  4. Generate TDL Scenario Figures.
  5. Copy all figures to the artifact directory.
"""

import os, sys, subprocess, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ARTIFACT_DIR = "/Users/apurva/.gemini/antigravity/brain/d2efda49-e66e-4b92-a08b-95db3cc3d4eb"
PY = sys.executable

def run(cmd, desc):
    print(f"\n{'='*72}")
    print(f"  STEP: {desc}")
    print(f"  CMD:  {cmd}")
    print(f"{'='*72}\n")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, cwd=ROOT)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"ERROR: Step '{desc}' failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"\n[OK] Step '{desc}' finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

def copy_artifacts():
    print(f"\n{'='*72}")
    print("  Copying generated figures to Artifacts Directory...")
    print(f"{'='*72}")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    figs = [
        'trust_vs_ncrn_full_pilot.png',
        'trust_vs_ncrn_sparse_pilot.png',
        'residual_noise_comparison.png',
        'nmse_tdl_64.png',
        'tdl_scenarios_LOS.png',
        'tdl_scenarios_NLOS.png',
        'tdl_scenarios_O2I.png',
        'tdl_scenarios_HST.png',
    ]
    for f in figs:
        src = os.path.join(ROOT, 'outputs', 'figures', f)
        if os.path.exists(src):
            dst = os.path.join(ARTIFACT_DIR, f)
            shutil.copy2(src, dst)
            print(f"  Copied {f} -> {dst}")

def main():
    start_time = time.time()
    print("*" * 72)
    print("  LAUNCHING FULL MATCHED TRUST-NET PIPELINE")
    print(f"  Python: {PY}")
    print(f"  Root:   {ROOT}")
    print("*" * 72)

    # 1. Training (30 epochs, 200 steps/epoch, bs=1, grad_accum=4 -> 6,000 samples)
    cmd_train = (
        f"{PY} stage2_tdl/trust_net/train_trust_net.py "
        f"--variant a1 --tag _v2 --k_dw 7 --epochs 30 --steps_per_epoch 200 "
        f"--bs 1 --grad_accum 4 --no_early_stopping"
    )
    run(cmd_train, "Train TRUST-Net V2 to Exact NCRN Parity (30 epochs, 6,000 samples)")

    # 2. Verification (Full-pilot & Sparse-pilot)
    cmd_eval = (
        f"{PY} stage2_tdl/trust_net/eval_trust_vs_ncrn.py "
        f"--trust_model models/trust_net_a1_v2_seed0.keras --samples 50 --eval_sparse"
    )
    run(cmd_eval, "Head-to-Head Verification (Full-Pilot & Sparse-Pilot)")

    # 3. Residual Noise Heatmaps
    cmd_residual = (
        f"{PY} stage2_tdl/trust_net/plot_residual_noise.py "
        f"--trust_model models/trust_net_a1_v2_seed0.keras --snr 10"
    )
    run(cmd_residual, "Residual Noise Spatial Heatmaps & Error Maps")

    # 4. TDL Scenario Figures
    cmd_scenarios = (
        f"{PY} plotting/plot_all_figures.py --quick"
    )
    run(cmd_scenarios, "Stage II Averaged & Doppler Scenario Benchmark Plots")

    # 5. Copy Artifacts
    copy_artifacts()

    total_time = time.time() - start_time
    print("\n" + "=" * 72)
    print(f"  ALL TASKS SUCCESSFULLY COMPLETED IN {total_time/60:.1f} MINUTES!")
    print("=" * 72)

if __name__ == '__main__':
    main()
