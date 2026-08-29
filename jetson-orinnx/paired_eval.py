"""jetson-orinnx/paired_eval.py -- HOST/Colab, deliberately NOT the Jetson (see
README.md's "Harness usage" section). This step is pure PyTorch -- no
TensorRT, no `trtexec`, nothing that actually touches the Jetson's SoC --
so it runs wherever is easiest, matching how the rest of this project
(E1-E5, P2.1, most of the baselines) already works. It needs
`out/replan/plans_hw.csv` (produced ON the Jetson by `replan.py`) copied
to wherever this runs -- see README.md's "Getting files off the Jetson"
section.

REGISTERED PROTOCOL (verbatim -- drafted for this deliverable following
this project's own established conventions for paired evaluation, fixed
manifests, and pre-registered gates; no separate external protocol
document was supplied at the time this was written, so this docstring IS
the registration. Any future change to the numbers below is a deviation
from this text and must be logged as such, same discipline as
experiments/phase2_2_baselines/deviations.md.):

  1. FIXED, DISJOINT DATA. A single `--imagenet-dir` (local, ImageFolder
     layout) is partitioned DETERMINISTICALLY by sorted file path: the
     first CALIB_N images are the calibration set (used only to compute
     ActivationQuantizer's amax_lookup, mirroring P2.1's own mechanism --
     see docstring note below on why this is a documented, independent
     recompute rather than an identifier-for-identifier match with
     P2.1's own HF-streamed calibration set); the NEXT PAIRED_EVAL_N
     images are the paired-eval set. These two sets never overlap by
     construction. The paired-eval set's manifest (sorted list of file
     paths) is hashed and recorded before any evaluation runs.
  2. PER BUDGET, PER IMAGE, THREE PLANS (third arm added on
     review, 2026-07 -- see replan.py's own module docstring and
     jetson-orinnx/README.md's "Joint vs staged under measured latency"
     section). For each of the 6 budget fractions, this script loads
     the BOPs-selected plan, the JOINT latency-selected plan, and the
     STAGED latency-selected plan (all three embedded directly in
     replan.py's plans_hw.csv -- `bops_plan_json`, `plan_json`,
     `staged_plan_json` columns respectively) for that budget, and
     evaluates ALL THREE on the IDENTICAL paired-eval images, in the
     same process, back to back, so no other source of variation
     (different images, different environment) can explain a
     difference. Two comparisons come out of this: BOPs vs JOINT (the
     originally registered "does the cost proxy matter" question) and
     JOINT vs STAGED (the thesis's actual headline claim -- "does joint
     beat staged" -- under real hardware cost, where
     experiments/phase1_E1-E5_E4a/E5_allocator_G1.py found it only TIED under simulated
     BOPs cost, "staging collapse").

     NOTE on budget definition (corrected, 2026-07, logged here
     per this section's own rule -- jetson-orinnx/ has no deviations.md, this
     docstring IS the registration): each budget's latency ceiling
     (`tau_b`, also in plans_hw.csv) is the MEASURED LATENCY of the
     BOPs-selected plan itself, not a fraction of the all-FP16 identity
     latency -- see replan.py's own module docstring for the full
     reasoning. This guarantees a feasible latency-selected plan exists
     at every one of the 6 budgets by construction (an earlier, since-
     corrected pairing did not -- see jetson-orinnx/README.md's Budget pairing
     section for that history), so this script evaluates all 6 budgets
     as originally intended, no skipping.
  3. PAIRED STATISTICS (two, independent). Per image, per budget:
     diff_bj = (joint_correct - bops_correct) and
     diff_js = (joint_correct - staged_correct), both in {-1, 0, +1}.
     Each gate statistic is the MEAN of its own diff pooled across all 6
     budgets x images (n = 6 x PAIRED_EVAL_N), with a 95% CI via the
     normal approximation to the paired mean (SE = std(diff, ddof=1) /
     sqrt(n)). Per-budget means are also reported for both, but each
     gate is evaluated on its own pooled statistic. These are two
     SEPARATE registered questions, not one combined test -- BOPs vs
     JOINT can pass or fail independently of JOINT vs STAGED.
  4. GATE (same registered criterion, applied independently to BOTH
     comparisons above -- not a new threshold invented for the second
     one). PASS iff pooled mean delta-top1 >= +0.4 PERCENTAGE POINTS AND
     the 95% CI excludes 0 (i.e. CI_low > 0). This is a symmetric,
     pre-registered, two-part gate -- both conditions are printed
     regardless of outcome, for both comparisons.
  5. NO PEEKING. No running/partial mean-delta or CI is printed or
     otherwise inspected before every (budget, image) pair has been
     evaluated -- only a plain image-count progress indicator is printed
     during the run. Both gate computations happen exactly once, after
     the full sweep, and their results (pass or fail) are written to
     paired_eval_result.json UNCONDITIONALLY -- a failing result is
     written and reported exactly like a passing one, never discarded or
     re-run silently.

PLAN-EXECUTION ENGINE: `_quantize_weight_rows`, `_quantize_tensor`,
`WeightQuantizer`, `ActivationQuantizer`, `ward_greedy_merge`,
`_assert_ward_theorem_gate`, `MergeScheduler`, `calibrate_activation_scales`,
`hash_state_dict`, `TARGET_ATTRS` below are a faithful, verbatim-where-
possible PORT of experiments/phase2_1_fleet/P2_tandem_pipeline.py's sections 8-11 -- not an
import (jetson-orinnx/ cannot import experiments/ code, see _common.py's module
docstring). This is the ONLY mechanism in this project that has been
validated to predict a plan's real top-1 accuracy, so it is reused here
by porting rather than re-derived from scratch.
"""
import argparse
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    compute_plan_divergence, ensure_out_dir, JETSON_ROOT, read_p2_csv_path, sha256_of_json_obj, write_json,
)

MODEL_NAME = "deit_small_patch16_224"
N_TOKENS_TOTAL = 197  # deit_small_patch16_224 -- same constant as replan.py, used only for the
# plan-divergence comparison below (compute_plan_divergence), not for any accuracy computation.
RESOLVED_TAG = "deit_small_patch16_224.fb_in1k"
TARGET_ATTRS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
BUDGET_FRACTIONS = (0.15, 0.21, 0.293, 0.41, 0.572, 0.80)
CALIB_N = 512
PAIRED_EVAL_N = 20000
BATCH_SIZE = 64
GATE_MEAN_THRESHOLD_PP = 0.4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_batch(batch_u8, device):
    """uint8 CHW -> normalized float32, applied per-BATCH at forward time, not on the full
    materialized ~20.5k-image working set. Real fix, 2026-07: the working set used to be
    materialized as already-normalized float32 (4 bytes/channel, ~11.5GB resident for
    CALIB_N+PAIRED_EVAL_N images), which was the actual cause of Colab RAM exhaustion --
    not something inherent to running this many images. uint8 storage (1 byte/channel, ~2.9GB
    resident) + per-batch normalization is the same pattern already proven in
    experiments/phase2_1_fleet/P2_tandem_pipeline.py's PREPROCESS/normalize_batch; ported here rather than
    re-derived (jetson-orinnx/ cannot import experiments/ code -- see module docstring)."""
    x = batch_u8.to(device, non_blocking=True).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


# ============================================================================
# Ported plan-execution engine (see module docstring for provenance)
# ============================================================================
def get_target_linear(model, block_idx, attr):
    mod = model.blocks[block_idx]
    for part in attr.split("."):
        mod = getattr(mod, part)
    return mod


def _quantize_weight_rows(W, bits):
    qmax = 2 ** (bits - 1) - 1
    w = W.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    w_hat = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w_hat.to(W.dtype)


def _quantize_tensor(x, scale, bits):
    qmax = 2 ** (bits - 1) - 1
    x_hat = torch.clamp(torch.round(x.float() / scale), -qmax, qmax) * scale
    return x_hat.to(x.dtype)


class WeightQuantizer:
    def __init__(self, model, block_bits):
        self.model, self.block_bits = model, block_bits
        self._linears, self._orig, self._quant = {}, {}, {}

    def __enter__(self):
        for block_idx, bits in self.block_bits.items():
            for attr in TARGET_ATTRS:
                lin = get_target_linear(self.model, block_idx, attr)
                key = (block_idx, attr)
                self._linears[key] = lin
                self._orig[key] = lin.weight.data
                self._quant[key] = _quantize_weight_rows(lin.weight.data, bits)
        return self

    def use_quantized(self):
        for key, lin in self._linears.items():
            lin.weight.data = self._quant[key]

    def use_original(self):
        for key, lin in self._linears.items():
            lin.weight.data = self._orig[key]

    def __exit__(self, *exc):
        self.use_original()


class ActivationQuantizer:
    def __init__(self, model, block_bits, amax_lookup):
        self.model, self.block_bits, self.amax_lookup = model, block_bits, amax_lookup
        self.active, self.handles = False, []

    def _make_hook(self, block_idx, attr, bits):
        qmax = 2 ** (bits - 1) - 1
        a_max = max(self.amax_lookup[(block_idx, attr)], 1e-12)
        scale = a_max / qmax

        def hook(module, args):
            if not self.active:
                return None
            return (_quantize_tensor(args[0], scale, bits),) + args[1:]

        return hook

    def __enter__(self):
        for block_idx, bits in self.block_bits.items():
            for attr in TARGET_ATTRS:
                lin = get_target_linear(self.model, block_idx, attr)
                self.handles.append(lin.register_forward_pre_hook(self._make_hook(block_idx, attr, bits)))
        return self

    def use_quantized(self):
        self.active = True

    def use_original(self):
        self.active = False

    def __exit__(self, *exc):
        self.active = False
        for h in self.handles:
            h.remove()
        self.handles = []


def ward_greedy_merge(features, sizes_init, r, track_membership=False):
    with torch.autocast(device_type=features.device.type, enabled=False):
        features = features.float()
        sizes_init = sizes_init.float()
        b, k, d = features.shape
        device = features.device
        centroids = features.clone()
        sizes = sizes_init.clone()
        assign = torch.arange(k, device=device).unsqueeze(0).repeat(b, 1) if track_membership else None

        sq_norms = (centroids**2).sum(-1)
        dot = torch.bmm(centroids, centroids.transpose(1, 2))
        sq_dist = (sq_norms.unsqueeze(2) + sq_norms.unsqueeze(1) - 2 * dot).clamp_min(0.0)
        size_prod = sizes.unsqueeze(2) * sizes.unsqueeze(1)
        size_sum = sizes.unsqueeze(2) + sizes.unsqueeze(1)
        cost = (size_prod / size_sum) * sq_dist
        diag = torch.eye(k, dtype=torch.bool, device=device).unsqueeze(0).expand(b, -1, -1)
        cost = cost.masked_fill(diag, float("inf"))

        cumulative_cost = torch.zeros(b, device=device)
        batch_idx = torch.arange(b, device=device)
        active = torch.ones(b, k, dtype=torch.bool, device=device)

        for _ in range(r):
            flat = cost.view(b, -1)
            min_idx = flat.argmin(dim=1)
            i, j = min_idx // k, min_idx % k

            s_i, s_j = sizes[batch_idx, i], sizes[batch_idx, j]
            s_new = s_i + s_j
            c_i, c_j = centroids[batch_idx, i], centroids[batch_idx, j]
            c_new = (s_i.unsqueeze(1) * c_i + s_j.unsqueeze(1) * c_j) / s_new.unsqueeze(1)
            merge_cost = cost[batch_idx, i, j]

            s_k = sizes
            old_ik, old_jk = cost[batch_idx, i, :], cost[batch_idx, j, :]
            denom = s_i.unsqueeze(1) + s_j.unsqueeze(1) + s_k
            new_row = (
                (s_i.unsqueeze(1) + s_k) * old_ik + (s_j.unsqueeze(1) + s_k) * old_jk - s_k * merge_cost.unsqueeze(1)
            ) / denom

            cumulative_cost = cumulative_cost + merge_cost
            centroids[batch_idx, i] = c_new
            sizes[batch_idx, i] = s_new
            sizes[batch_idx, j] = 0.0
            active[batch_idx, j] = False
            if track_membership:
                match = assign == j.unsqueeze(1)
                assign = torch.where(match, i.unsqueeze(1).expand(-1, k), assign)

            cost[batch_idx, i, :] = new_row
            cost[batch_idx, :, i] = new_row
            cost[batch_idx, i, i] = float("inf")
            cost[batch_idx, j, :] = float("inf")
            cost[batch_idx, :, j] = float("inf")

        order = torch.argsort((~active).float(), dim=1, stable=True)
        keep = order[:, : k - r]
        new_patches = torch.gather(centroids, 1, keep.unsqueeze(-1).expand(-1, -1, d))
        new_sizes = torch.gather(sizes, 1, keep)
        return {"patches": new_patches, "sizes": new_sizes, "cost": cumulative_cost,
                "centroids": centroids, "assign": assign, "active": active}


def _assert_ward_theorem_gate(pre_merge_features, pre_merge_sizes, ward_result, rel_tol=1e-4):
    assign, centroids = ward_result["assign"], ward_result["centroids"]
    b, k, d = pre_merge_features.shape
    gathered = torch.gather(centroids, 1, assign.unsqueeze(-1).expand(-1, -1, d))
    true_sse = (pre_merge_sizes.unsqueeze(-1) * (pre_merge_features - gathered) ** 2).sum(dim=(1, 2))
    cum_cost = ward_result["cost"]
    rel_err = (true_sse - cum_cost).abs() / cum_cost.clamp_min(1e-12)
    max_rel_err = rel_err.max().item()
    assert max_rel_err < rel_tol, f"Ward theorem gate FAILED: max_rel_err={max_rel_err:.3e} >= {rel_tol:.0e}"
    return max_rel_err


class MergeScheduler:
    def __init__(self, model, points, device):
        self.model, self.points, self.device = model, points, device
        self.handles, self.sizes, self.gate_d_rel_errs = [], None, []

    def _make_hook(self, point, is_first_point):
        def hook(module, inp, out):
            cls_tok, patches = out[:, :1, :], out[:, 1:, :]
            b, n_cur, d = patches.shape
            if is_first_point:
                self.sizes = torch.ones(b, n_cur, device=self.device)
            r = point["r"]
            feats_fp32 = patches.float()
            sizes_before = self.sizes
            result = ward_greedy_merge(feats_fp32, sizes_before, r, track_membership=True)
            rel_err = _assert_ward_theorem_gate(feats_fp32, sizes_before, result)
            self.gate_d_rel_errs.append(rel_err)
            new_patches, new_sizes = result["patches"], result["sizes"]
            self.sizes = new_sizes
            new_out = torch.cat([cls_tok, new_patches.to(out.dtype)], dim=1)
            return new_out

        return hook

    def __enter__(self):
        for idx, point in enumerate(self.points):
            h = self.model.blocks[point["block_idx"]].register_forward_hook(self._make_hook(point, idx == 0))
            self.handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles, self.sizes, self.gate_d_rel_errs = [], None, []


def calibrate_activation_scales(model, images, batch_size, n_blocks, device):
    amax_running = {}

    def make_hook(block_idx, attr):
        def hook(module, args):
            batch_max = args[0].float().abs().max().item()
            key = (block_idx, attr)
            amax_running[key] = max(amax_running.get(key, 0.0), batch_max)
        return hook

    handles = [get_target_linear(model, b, attr).register_forward_pre_hook(make_hook(b, attr))
               for b in range(n_blocks) for attr in TARGET_ATTRS]
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], batch_size):
                model(normalize_batch(images[start:start + batch_size], device))
    finally:
        for h in handles:
            h.remove()
    return amax_running


def hash_state_dict(model) -> str:
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def plan_to_block_bits(plan):
    return {int(l) - 1: bits for l, bits in plan["bits"].items() if bits != 16}


def plan_to_merge_points(plan):
    return [{"block_idx": m["block"] - 1, "r": m["r"]} for m in sorted(plan["merges"], key=lambda m: m["block"])]


def evaluate_plan_top1(model, images, labels, plan, amax_lookup, batch_size, device):
    """Leaner sibling of P2.1's run_plan: only per-image top1 correctness
    is needed for paired_eval.py's gate, not the full MSE/KL/flip metric
    set run_plan also computes -- everything else about the mechanism
    (order of context managers, weight-restoration assertion) is
    unchanged from the port."""
    block_bits = plan_to_block_bits(plan)
    merge_points = plan_to_merge_points(plan)
    n = images.shape[0]
    correct = torch.zeros(n, dtype=torch.bool)

    weight_q = WeightQuantizer(model, block_bits) if block_bits else None
    act_q = ActivationQuantizer(model, block_bits, amax_lookup) if block_bits else None
    sched = MergeScheduler(model, merge_points, device) if merge_points else None

    pre_hash = hash_state_dict(model)
    with ExitStack() as stack:
        if weight_q is not None:
            stack.enter_context(weight_q)
            weight_q.use_quantized()
        if act_q is not None:
            stack.enter_context(act_q)
            act_q.use_quantized()
        if sched is not None:
            stack.enter_context(sched)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = normalize_batch(images[start:start + batch_size], device)
                lbl = labels[start:start + batch_size].to(device)
                logits = model(batch)
                correct[start:start + batch.shape[0]] = (logits.argmax(dim=1) == lbl).cpu()

    post_hash = hash_state_dict(model)
    assert post_hash == pre_hash, "Plan execution FAILED to restore model weights exactly."
    return correct.numpy()


# ============================================================================
# Data loading (local directory, deterministic disjoint split -- see module docstring item 1)
# ============================================================================
def load_partitioned_dataset(imagenet_dir: str):
    import torchvision.datasets as tvd
    import torchvision.transforms as T

    # PILToTensor (uint8, unnormalized) -- NOT ToTensor+Normalize. Real fix, 2026-07: see
    # normalize_batch's docstring above -- normalization is applied per-batch at forward time,
    # not baked into the ~20.5k-image resident working set.
    tf = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.PILToTensor(),
    ])
    ds = tvd.ImageFolder(imagenet_dir, transform=tf)
    order = sorted(range(len(ds)), key=lambda i: ds.samples[i][0])
    assert len(order) >= CALIB_N + PAIRED_EVAL_N, (
        f"{imagenet_dir} has {len(order)} images, need >= {CALIB_N + PAIRED_EVAL_N} "
        f"({CALIB_N} calibration + {PAIRED_EVAL_N} paired-eval, disjoint)."
    )
    calib_idx = order[:CALIB_N]
    eval_idx = order[CALIB_N:CALIB_N + PAIRED_EVAL_N]
    assert set(calib_idx).isdisjoint(eval_idx), "calibration/paired-eval split is not disjoint -- bug"

    eval_paths = sorted(ds.samples[i][0] for i in eval_idx)
    manifest_hash = sha256_of_json_obj(eval_paths)

    def load_subset(indices, desc):
        # Preallocated + filled slot-by-slot, not a Python list + torch.stack -- avoids briefly
        # holding both the per-image list AND the final stacked tensor in RAM at once. Same
        # pattern as experiments/phase2_1_fleet/P2_tandem_pipeline.py's _materialize.
        n = len(indices)
        imgs = torch.empty((n, 3, 224, 224), dtype=torch.uint8)
        labels = torch.empty(n, dtype=torch.long)
        for out_i, ds_i in enumerate(indices):
            img, label = ds[ds_i]
            imgs[out_i] = img
            labels[out_i] = label
            if (out_i + 1) % 2000 == 0:
                print(f"  {desc}: {out_i + 1}/{n}...")
        return imgs, labels

    calib_images, _ = load_subset(calib_idx, "decoding calibration images")
    eval_images, eval_labels = load_subset(eval_idx, "decoding paired-eval images")
    return calib_images, eval_images, eval_labels, manifest_hash


# ============================================================================
# Paired statistics
# ============================================================================
def paired_ci(diffs_pp: np.ndarray, confidence=0.95):
    n = len(diffs_pp)
    mean = float(np.mean(diffs_pp))
    se = float(np.std(diffs_pp, ddof=1) / np.sqrt(n))
    z = 1.959963984540054  # 95% two-sided normal
    return mean, mean - z * se, mean + z * se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TF32 GUARD: this script runs on whatever GPU host is available (Colab, a university GPU,
    # or the Jetson itself) -- P2.1's own reference numbers were measured on a Tesla T4 (Turing,
    # no TF32 hardware path at all), but many hosts this could run on (Colab's A100/L4, the
    # Jetson's own Ampere GPU) DO support TF32. Left enabled, TF32 would silently round fp32
    # matmuls (WeightQuantizer, ward_greedy_merge's bmm) differently depending on which host
    # happened to run this, a real cross-hardware numerical discrepancy this project's own
    # discipline (see deviations.md's determinism entries) does not silently allow. Disabled
    # explicitly and unconditionally, regardless of host, verified by self_test.py's TF32 check.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"=== paired_eval.py: device={device} (TF32 explicitly disabled -- see comment above) ===")

    print(f"Loading + partitioning {args.imagenet_dir} ({CALIB_N} calib + {PAIRED_EVAL_N} paired-eval, disjoint)...")
    calib_images, eval_images, eval_labels, manifest_hash = load_partitioned_dataset(args.imagenet_dir)
    print(f"Paired-eval manifest hash: {manifest_hash}")

    import timm
    try:
        model = timm.create_model(MODEL_NAME, pretrained=True)
    except Exception:
        model = timm.create_model(RESOLVED_TAG, pretrained=True)
    model = model.eval().to(device).requires_grad_(False)
    n_blocks = len(model.blocks)
    checkpoint_hash = hash_state_dict(model)
    print(f"Loaded {RESOLVED_TAG}, checkpoint_hash={checkpoint_hash}")

    print(f"Calibrating activation scales from {CALIB_N} local images "
          "(independent recompute vs P2.1's own HF-streamed calibration set -- see module docstring item 1)...")
    amax_lookup = calibrate_activation_scales(model, calib_images, BATCH_SIZE, n_blocks, device)

    # JETSON_ROOT-relative, not CWD-relative -- same bug class found and fixed elsewhere in
    # this folder, 2026-07 (see jetson-orinnx/README.md's INT8 findings section) -- fixed here too.
    hw_path = JETSON_ROOT / "out" / "replan" / "plans_hw.csv"
    assert hw_path.exists(), f"{hw_path} not found -- run replan.py first."
    hw_df = pd.read_csv(hw_path)
    assert len(hw_df) == len(BUDGET_FRACTIONS), (
        f"{hw_path} has {len(hw_df)} rows, expected {len(BUDGET_FRACTIONS)} -- re-run replan.py."
    )

    # Lightweight independent cross-check: plans_hw.csv's own embedded bops_plan_json should
    # exactly match P2.1's real committed plans.csv (replan.py derives it from there in the
    # first place -- this just confirms the embedding round-tripped correctly, catching a real
    # bug rather than a hardware finding if it ever doesn't).
    bops_df = pd.read_csv(read_p2_csv_path(MODEL_NAME, "plans.csv"))
    assert len(bops_df) == len(BUDGET_FRACTIONS), "P2.1 plans.csv row count mismatch"
    for i in range(len(BUDGET_FRACTIONS)):
        assert json.loads(hw_df.iloc[i]["bops_plan_json"]) == json.loads(bops_df.iloc[i]["plan_json"]), (
            f"budget index {i}: plans_hw.csv's embedded bops_plan_json does not match P2.1's "
            "own plans.csv -- replan.py's plan embedding disagrees with its own source, halting."
        )

    # Budgets are feasible at all 6 by construction (tau_b = the BOPs plan's own measured
    # latency -- see replan.py's module docstring's "BUDGET PAIRING" section) -- no skipping.
    # THREE plans per budget (staged arm added on review, 2026-07 -- see module
    # docstring item 2 and replan.py's own docstring).
    per_budget_results = []
    all_diffs_bj = []  # BOPs vs JOINT -- the originally registered comparison
    all_diffs_js = []  # JOINT vs STAGED -- the thesis's central claim, under real hardware cost
    for i, budget_frac in enumerate(BUDGET_FRACTIONS):
        bops_plan = json.loads(hw_df.iloc[i]["bops_plan_json"])
        joint_plan = json.loads(hw_df.iloc[i]["plan_json"])
        staged_plan = json.loads(hw_df.iloc[i]["staged_plan_json"])
        tau_ms = float(hw_df.iloc[i]["tau_ms"])
        print(f"\n=== budget_fraction={budget_frac:.1%} ({i + 1}/{len(BUDGET_FRACTIONS)}), tau_b={tau_ms:.4f}ms ===")
        print(f"  BOPs plan:   bits={bops_plan['bits']} merges={bops_plan['merges']}")
        print(f"  JOINT plan:  bits={joint_plan['bits']} merges={joint_plan['merges']}")
        print(f"  STAGED plan: bits={staged_plan['bits']} merges={staged_plan['merges']}")

        t0 = time.perf_counter()
        bops_correct = evaluate_plan_top1(model, eval_images, eval_labels, bops_plan, amax_lookup, BATCH_SIZE, device)
        joint_correct = evaluate_plan_top1(model, eval_images, eval_labels, joint_plan, amax_lookup, BATCH_SIZE, device)
        staged_correct = evaluate_plan_top1(model, eval_images, eval_labels, staged_plan, amax_lookup, BATCH_SIZE, device)
        elapsed = time.perf_counter() - t0
        # NO PEEKING: no mean/delta printed here -- see module docstring item 5. Progress only.
        print(f"  evaluated {len(eval_images)} paired images for all three plans in {elapsed:.1f}s")

        diff_bj_pp = (joint_correct.astype(np.float64) - bops_correct.astype(np.float64)) * 100.0
        diff_js_pp = (joint_correct.astype(np.float64) - staged_correct.astype(np.float64)) * 100.0
        all_diffs_bj.append(diff_bj_pp)
        all_diffs_js.append(diff_js_pp)

        # Pre-flight patch item 2: carry the SAME plan-divergence fields replan.py writes into
        # plans_hw_divergence.json, computed here directly (all three plan dicts already
        # loaded) via the shared _common.compute_plan_divergence -- see that function's
        # docstring for why this is NOT the same as a raw dict-equality comparison.
        bj_identical, bj_n_diff, bj_per_block_diff = compute_plan_divergence(bops_plan, joint_plan, N_TOKENS_TOTAL)
        js_identical, js_n_diff, js_per_block_diff = compute_plan_divergence(joint_plan, staged_plan, N_TOKENS_TOTAL)
        per_budget_results.append({
            "budget_fraction": budget_frac, "tau_ms": tau_ms,
            "bops_plan": bops_plan, "joint_plan": joint_plan, "staged_plan": staged_plan,
            "bops_vs_joint_identical": bj_identical, "bops_vs_joint_n_blocks_differing": bj_n_diff,
            "bops_vs_joint_per_block_diff": bj_per_block_diff,
            "joint_vs_staged_identical": js_identical, "joint_vs_staged_n_blocks_differing": js_n_diff,
            "joint_vs_staged_per_block_diff": js_per_block_diff,
            "bops_top1_pct": float(bops_correct.mean() * 100), "joint_top1_pct": float(joint_correct.mean() * 100),
            "staged_top1_pct": float(staged_correct.mean() * 100),
            "mean_diff_bj_pp": float(diff_bj_pp.mean()), "mean_diff_js_pp": float(diff_js_pp.mean()),
        })

    # --- TWO pooled gate statistics, each computed exactly once, after the full sweep, all 6
    # budgets -- see module docstring item 3/4: two separate registered questions, same gate
    # criterion applied independently to each.
    pooled_diffs_bj = np.concatenate(all_diffs_bj)
    mean_bj_pp, ci_bj_low, ci_bj_high = paired_ci(pooled_diffs_bj)
    gate_bj_mean_ok = mean_bj_pp >= GATE_MEAN_THRESHOLD_PP
    gate_bj_ci_ok = ci_bj_low > 0.0
    gate_bj_pass = bool(gate_bj_mean_ok and gate_bj_ci_ok)

    pooled_diffs_js = np.concatenate(all_diffs_js)
    mean_js_pp, ci_js_low, ci_js_high = paired_ci(pooled_diffs_js)
    gate_js_mean_ok = mean_js_pp >= GATE_MEAN_THRESHOLD_PP
    gate_js_ci_ok = ci_js_low > 0.0
    gate_js_pass = bool(gate_js_mean_ok and gate_js_ci_ok)

    all_plans_identical = all(r["bops_vs_joint_identical"] for r in per_budget_results)
    all_joint_staged_identical = all(r["joint_vs_staged_identical"] for r in per_budget_results)
    print("\n=== FINAL RESULT (computed once, after the full sweep -- see module docstring item 5) ===")
    print("--- Comparison 1: BOPs-selected vs JOINT (does the cost proxy matter?) ---")
    print(f"Pooled mean paired delta-top1 (joint - bops): {mean_bj_pp:+.4f}pp  95% CI=[{ci_bj_low:+.4f}, {ci_bj_high:+.4f}]pp")
    print(f"Gate: mean >= +{GATE_MEAN_THRESHOLD_PP}pp: {gate_bj_mean_ok}   CI excludes 0: {gate_bj_ci_ok}   "
          f"=> GATE {'PASS' if gate_bj_pass else 'FAIL'}")
    print("\n--- Comparison 2: JOINT vs STAGED (does joint beat staged under real hardware cost?) ---")
    print(f"Pooled mean paired delta-top1 (joint - staged): {mean_js_pp:+.4f}pp  95% CI=[{ci_js_low:+.4f}, {ci_js_high:+.4f}]pp")
    print(f"Gate: mean >= +{GATE_MEAN_THRESHOLD_PP}pp: {gate_js_mean_ok}   CI excludes 0: {gate_js_ci_ok}   "
          f"=> GATE {'PASS' if gate_js_pass else 'FAIL'}")
    print(f"(G1/E5 found JOINT and STAGED tied EXACTLY under simulated BOPs cost -- \"staging "
          f"collapse\", plans_identical=True at every budget. Under real measured latency: "
          f"plans_identical at {sum(1 for r in per_budget_results if r['joint_vs_staged_identical'])}"
          f"/{len(per_budget_results)} budgets.)")
    print("\n--- Per-budget breakdown ---")
    for r in per_budget_results:
        print(f"  budget={r['budget_fraction']:.1%}: bops_top1={r['bops_top1_pct']:.4f}% "
              f"joint_top1={r['joint_top1_pct']:.4f}% staged_top1={r['staged_top1_pct']:.4f}%  "
              f"diff(joint-bops)={r['mean_diff_bj_pp']:+.4f}pp diff(joint-staged)={r['mean_diff_js_pp']:+.4f}pp  "
              f"bops_vs_joint_identical={r['bops_vs_joint_identical']} "
              f"joint_vs_staged_identical={r['joint_vs_staged_identical']}")
    if all_plans_identical:
        print("\nINTERPRETATION NOTE (BOPs vs JOINT): plans_identical=True at every budget -- the "
              "BOPs-selected and JOINT plans never diverged. A near-zero pooled delta above would "
              "therefore mean 'hardware cost was effectively additive on this device' (the two cost "
              "axes always agreed on which plan to pick), NOT 'joint allocation does not help' -- "
              "those are different findings and this run cannot distinguish them without a "
              "divergent budget.")
    else:
        print(f"\nINTERPRETATION NOTE (BOPs vs JOINT): plans diverged at "
              f"{sum(1 for r in per_budget_results if not r['bops_vs_joint_identical'])}/"
              f"{len(per_budget_results)} budgets. PRE-REGISTERED INTERPRETATION (locked in "
              "writing BEFORE this result was known, reviewed, 2026-07 -- see "
              "jetson-orinnx/README.md's Budget pairing section): at this run's budgets, P2.1's own "
              "BOPs-selected plans mostly cost MORE real latency than doing nothing at all "
              "(tau_b exceeds Latency_id at most budgets -- see replan.py's own printed "
              "'Proxy divergence' table). Any positive delta above therefore means THE "
              "LATENCY-AWARE PLANNER WINS BY DECLINING TO COMPRESS, because BOPs-selected "
              "compression costs more latency than it saves on this device. This is a "
              "PROXY-DIVERGENCE result (does the cost proxy matter -- yes, substantially), "
              "NOT an ALLOCATION-QUALITY result (it does not show joint allocation choosing "
              "well UNDER a real constraint, since at most of these budgets there effectively "
              "isn't one). DO NOT report the gate number without this sentence.")
    if all_joint_staged_identical:
        n_joint_is_identity = sum(
            1 for r in per_budget_results
            if r["joint_plan"]["merges"] == [] and all(b == 16 for b in r["joint_plan"]["bits"].values())
        )
        print("\nINTERPRETATION NOTE (JOINT vs STAGED): plans_identical=True at every budget. "
              "CORRECTED framing (reviewed, 2026-07 -- do NOT report this as \"staging "
              "collapse persists\", an earlier mischaracterization by this same pipeline): "
              f"{n_joint_is_identity}/{len(per_budget_results)} of these budgets have JOINT "
              "converging on the literal all-FP16, no-merge identity plan -- looser than doing "
              "nothing, so every selection rule trivially agrees there. Record instead: \"not "
              "tested -- at these budgets the latency-optimal plan is identity, so all "
              "selection rules coincide trivially.\" Joint vs staged under real hardware cost "
              "remains UNTESTED -- see jetson-orinnx/README.md's \"Joint vs staged\" section for the "
              "planned follow-up sweep (budgets set directly inside the achievable range, "
              "forcing real compression).")
    else:
        print(f"\nINTERPRETATION NOTE (JOINT vs STAGED): plans diverged at "
              f"{sum(1 for r in per_budget_results if not r['joint_vs_staged_identical'])}/"
              f"{len(per_budget_results)} budgets -- a genuine, non-degenerate joint-vs-staged "
              "separation under real hardware cost. The pooled delta above DOES reflect that "
              "allocation difference, not just measurement noise on identical plans.")

    result = {
        "model_name": MODEL_NAME, "checkpoint_hash": checkpoint_hash, "manifest_hash": manifest_hash,
        "calib_n": CALIB_N, "paired_eval_n": PAIRED_EVAL_N, "n_budgets": len(BUDGET_FRACTIONS),
        "n_pooled_pairs": len(pooled_diffs_bj),  # same for both -- both pool over all 6 budgets x PAIRED_EVAL_N
        "comparison_bops_vs_joint": {
            "description": "Does the cost proxy matter? (originally registered comparison)",
            "pre_registered_interpretation": (
                "Locked in writing BEFORE this result was known, reviewed, 2026-07 -- "
                "see jetson-orinnx/README.md's Budget pairing section. At this run's budgets, P2.1's "
                "own BOPs-selected plans mostly cost MORE real latency than doing nothing at "
                "all (tau_b exceeds Latency_id at most budgets). A positive delta here means "
                "the latency-aware planner wins by DECLINING TO COMPRESS, because BOPs-selected "
                "compression costs more latency than it saves on this device -- a "
                "PROXY-DIVERGENCE result, not an ALLOCATION-QUALITY result. Must not be "
                "reported without this sentence."
            ),
            "pooled_mean_diff_pp": mean_bj_pp, "pooled_ci_low_pp": ci_bj_low, "pooled_ci_high_pp": ci_bj_high,
            "gate_mean_threshold_pp": GATE_MEAN_THRESHOLD_PP,
            "gate_mean_ok": gate_bj_mean_ok, "gate_ci_ok": gate_bj_ci_ok, "gate_pass": gate_bj_pass,
            "all_plans_identical": all_plans_identical,
        },
        "comparison_joint_vs_staged": {
            "description": (
                "Does joint beat staged under real hardware cost? (thesis's central claim -- "
                "added on review, 2026-07)"
            ),
            "status_note": (
                "CORRECTED, reviewed, 2026-07: this run does NOT re-test "
                "experiments/phase1_E1-E5_E4a/E5_allocator_G1.py's 'staging collapse' finding, and must not be "
                "reported as such. At budgets where JOINT converges on the literal identity "
                "plan (tau_b >= Latency_id), every selection rule trivially agrees -- a "
                "degenerate coincidence, not a joint-vs-staged result. Correct framing: 'not "
                "tested -- at these budgets the latency-optimal plan is identity, so all "
                "selection rules coincide trivially.' See per_budget entries and "
                "jetson-orinnx/README.md's 'Joint vs staged' section for the planned follow-up sweep "
                "(budgets set directly inside the achievable range) that actually forces "
                "compression and makes this a live test."
            ),
            "pooled_mean_diff_pp": mean_js_pp, "pooled_ci_low_pp": ci_js_low, "pooled_ci_high_pp": ci_js_high,
            "gate_mean_threshold_pp": GATE_MEAN_THRESHOLD_PP,
            "gate_mean_ok": gate_js_mean_ok, "gate_ci_ok": gate_js_ci_ok, "gate_pass": gate_js_pass,
            "all_plans_identical": all_joint_staged_identical,
        },
        "per_budget": per_budget_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = ensure_out_dir("paired_eval") / "paired_eval_result.json"
    write_json(out_path, result)  # written UNCONDITIONALLY -- pass/fail either way, see module docstring item 5
    print(f"\nWrote {out_path} (BOPs-vs-JOINT gate {'PASS' if gate_bj_pass else 'FAIL'}, "
          f"JOINT-vs-STAGED gate {'PASS' if gate_js_pass else 'FAIL'} -- written either way).")


if __name__ == "__main__":
    main()
