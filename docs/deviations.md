# Deviations from published protocols

**A note on paths in older entries below**: this file was written
incrementally over the project's development, before the public-release
directory reorganization. Entries written before that reorg refer to the
pre-reorg layout (`baselines/` for what is now `experiments/phase2_2_baselines/`,
`notebooks/` for what is now split across `experiments/phase1_E1-E5_E4a/`
and `experiments/phase2_1_fleet/`) -- left as originally written where the
reference is describing historical debugging narrative (e.g. a specific
past bug's exact cwd/path mechanics), rather than rewritten to describe a
directory structure that didn't exist at the time. See `README.md`'s
repository map for the current layout.

Per the fairness rule, every baseline runs at its documented settings only
(no tuning). This file records every place where reproducing that exactly
was not possible or not fully verifiable, and why -- so any number in
`results.csv` can be traced back to whether it is a clean reproduction or
carries a caveat. Updated as B1-B4 are actually run on the university GPU;
the entries below are the deviations already known from the repos'
documentation, filled in before any run.

## Structural (do not affect measured numbers)

1. **Four separate Python environments, not one shared `requirements.txt`.**
   The three upstream repos' OWN documented (or, where undocumented,
   verified-compatible) pins are mutually incompatible, confirmed by
   checking each repo directly (none ships a `requirements.txt`; all three
   document install via README prose only):
   - ToMe (`INSTALL.md`): `torch >= 1.12.1` (needed for `scatter_reduce`),
     `timm == 0.4.12`.
   - FQ-ViT (`README.md`): `conda install pytorch=1.7.1 torchvision
     cudatoolkit=10.1` -- an EXACT pin that directly conflicts with ToMe's
     `>=1.12.1` floor. There is no single torch version satisfying both.
   - RepQ-ViT (`README.md`): only pins `timm == 0.4.12` (recommended);
     does not pin torch explicitly.
   Forcing all four baselines into one environment is therefore not just
   risky but IMPOSSIBLE at the repos' own documented pins. Each baseline
   gets its own `requirements.txt` (`b1_tome/`, `b2_repq_vit/`,
   `b3_fq_vit/`, `b4_sequential_composites/`); the top-level
   `experiments/phase2_2_baselines/requirements.txt` covers only the framework-free
   orchestration code (`common/bops.py`, `common/results_io.py`,
   `common/repro_gate.py`, `common/repo_utils.py`, `compare_pareto.py`),
   which has zero torch/timm dependency specifically so it works
   identically regardless of which baseline venv last touched
   `results.csv`.
   **Operational risk, not resolved here:** FQ-ViT's `cudatoolkit=10.1` pin
   predates most current GPU driver stacks; on the actual university GPU
   box this will likely need a newer CUDA build of a similar torch 1.7.x
   (or a tolerance test against a newer torch) -- a hardware-dependent
   judgment call for whoever runs B3, recorded in `b3_fq_vit/requirements.txt`'s
   header comment rather than guessed here.
   ToMe's `timm==0.4.12` pin also directly patches `timm.models.
   vision_transformer.Block`/`Attention` internals (see `b1_tome/run.py`'s
   docstring) -- B1 and B4 use that exact pinned timm version rather than
   TANDEM's own modern one, since ToMe's patch was written against, and
   only tested against, 0.4.12's internal Block/Attention structure.
2. **Local ImageNet-1k ImageFolder, not HF streaming.** Phase 1/P2 (Colab)
   streamed ImageNet-1k from the HF Hub. RepQ-ViT's and FQ-ViT's own
   `test_quant.py` scripts expect a standard local
   `<root>/{train,val}/<wnid>/*.JPEG` ImageFolder layout and cannot be
   pointed at a streaming source without modifying their code (which the
   fairness rule prohibits). All four baselines therefore require a local
   ImageNet-1k copy on the university GPU box (`IMAGENET_DIR` env var);
   `common/imagenet.py` uses `torchvision.datasets.ImageFolder` against
   `IMAGENET_DIR/val`, not the notebooks' HF loader. The full standard val
   split is exactly 50,000 images, so no subsetting logic is needed here
   (unlike P2's streamed 50k pull).

## B1 (ToMe) — reproduction-gate coverage

3. **[CORRECTED -- see item #31] No off-the-shelf DeiT-S reference
   exists anywhere in ToMe's README or paper.** Originally this entry
   claimed a "published DeiT-S off-the-shelf reference" of 79.36% at r=13
   and gated our off-the-shelf sweep against it -- WRONG, a setting
   mismatch, not a real reproduction target (see item #31 for the full
   correction). The ToMe README's DeiT table (`DeiT-S/16, r=13 -> 79.36%`,
   baseline 79.8%) is the TRAINED-with-ToMe setting (ships a checkpoint,
   `deit_S_r13.pth`), not off-the-shelf. Confirmed by fetching the README
   at the pinned commit directly: the actual off-the-shelf table (caption
   "expected results when using the timm implementation off-the-shelf")
   covers ViT-S/B/L only, no DeiT rows at all. The paper's appendix
   (arXiv:2210.09461, Table 11 "Full DeiT Results") was also checked and
   is trained-only too ("Results from running a standard 300 epoch DeiT-S
   ... training run with the official codebase"). **No off-the-shelf
   DeiT-S r-sweep exists anywhere in ToMe's own materials.** r=8, r=13
   (off_the_shelf setting), and r=16 are therefore all reported with
   `gate_status=no_published_reference`, never gated -- see item #31 for
   what IS gated instead.
4. **`prop_attn=True` for DeiT**, per the README's explicit instruction
   ("instantiate a DeiT timm model, patch it with the timm patch
   (`prop_attn=True`)"). This differs from the generic non-DeiT ViT example
   in the same README (which omits `prop_attn`) -- confirmed from
   `tome/patch/timm.py`'s `apply_patch(model, trace_source=False,
   prop_attn=True)` signature that `prop_attn` only changes attention-score
   biasing by log-token-size, not the merge count itself, so this is a
   faithful application of their documented per-architecture default, not a
   tuning choice.

## B2 (RepQ-ViT) — reproduction-gate coverage

5. **No published W8A8 reference located.** RepQ-ViT's
   `classification/README.md` results table publishes W4/A4 and W6/A6 only.
   **Gate applied at W6A6 only** (DeiT-S 78.90%, DeiT-B 81.27%, ViT-B
   83.62%, all ±0.3%). W8A8 is still run (their `test_quant.py` accepts
   arbitrary `--w_bits`/`--a_bits`) and reported, `gate_status=
   no_published_reference`.
6. **CLI flag names verified from source, not the README prose.** The
   `classification/README.md` usage snippet shows `--w_bit`/`--a_bit`
   (singular); the actual `test_quant.py` (pinned commit) defines
   `--w_bits`/`--a_bits` (plural). The runner uses the verified-from-source
   flag names.
7. **Calibration is exactly one batch of 32 images** drawn from the
   training split (`--calib-batchsize 32`, their default) -- confirmed from
   `test_quant.py` source, matching the spec's "32-image calibration per
   their protocol" exactly. Not overridden.

## B3 (FQ-ViT) — reproduction-gate coverage

8. **Calibration is their own default (100 x 10 = 1,000 images)**, distinct
   from RepQ-ViT's 32 -- each baseline uses ITS OWN documented default per
   the "no tuning" rule; the two are not made consistent with each other.
9. **"Ours" configuration** (`--quant --ptf --lis --quant-method minmax`) is
   used, matching the README's best-row (8/8/8: DeiT-S 79.17%, DeiT-B
   81.20%, ViT-B 83.31%) -- the paper's own proposed method, not a plain
   MinMax baseline from the same table (that would be a different,
   weaker-by-design comparison point, not "FQ-ViT" as a method).

## B4 (sequential composites) — highest-risk runner, flagged explicitly

10. **This is new integration code.** Neither ToMe nor RepQ-ViT is designed
    to compose with the other; `b4_sequential_composites/run.py` imports
    ToMe's `patch.timm` and RepQ-ViT's `quant`/`utils` modules as libraries
    and replicates `test_quant.py`'s calibration -> scale-reparameterization
    -> re-calibration -> validate sequence (verified against the pinned
    commit's actual source), inserting the ToMe patch between RepQ-ViT's
    `build_model` and `quant_model` steps so calibration and evaluation both
    run on the ALREADY-MERGED model, per spec. **This glue code has not been
    executed** (no GPU in the environment that wrote it) -- treat the first
    real run as a validation pass, not a trusted number, until it produces
    non-degenerate (non-NaN, non-collapsed) accuracy. The load-bearing
    assumption: RepQ-ViT's scale-reparameterization step walks
    `q_model.blocks.named_modules()` expecting standard `norm1`/`norm2`/
    `attn.qkv`/`mlp.fc1` submodule names -- ToMe's patch only reassigns
    `__class__` and overrides `forward` on `Block`/`Attention`, leaving the
    submodule tree unchanged, so this should still resolve correctly, but
    it is asserted (not just assumed) in the runner before proceeding.
11. **No published reference exists for either composite** (they are not a
    result either upstream paper reports) -- B4 has no reproduction gate by
    construction; both rows are `gate_status=not_applicable`.

## Host-equivalence audit (P2.2 finalization) — new deviations introduced

12. **Exact `==` version pins are NOT individually execution-verified.**
    torch/torchvision pins are real, verified OFFICIAL companion release
    pairs (confirmed via matching PyPI upload timestamps: torch==1.7.1 /
    torchvision==0.8.2 for FQ-ViT, torch==1.12.1 / torchvision==0.13.1
    for ToMe/RepQ-ViT/B4 -- the latter chosen as ToMe's own documented
    floor, the version closest to what its authors tested against, not
    an arbitrary newer release). numpy/scipy/tqdm/Pillow are pinned to
    releases contemporaneous with the chosen torch version (same
    reasoning: closer to what each repo's authors would have used) but
    were selected via PyPI release-date research, not by actually
    installing and running each repo's code (no GPU in this environment)
    -- the host-equivalence gate (`common/equivalence_gate.py`) is the
    real, empirical verification mechanism once these run for real; a
    `requirements_hash` mismatch there prints a full `pip freeze` diff.
13. **`CUBLAS_WORKSPACE_CONFIG=':4096:8'` is set for BOTH B2 and B3's
    subprocess**, via `common.determinism.subprocess_env_with_tf32_disabled`.
    FQ-ViT's own `seed()` already sets this (verified from source);
    RepQ-ViT's does NOT. Setting it for RepQ-ViT too is a
    determinism-only addition (it can only change WHICH cuBLAS algorithm
    variant gets selected for reproducibility, never the transform/
    calibration-count/bit-config protocol) -- not a fairness-rule
    violation, but flagged here as a real, intentional difference from
    RepQ-ViT's own unmodified defaults.
14. **B2/B3 have no batch-level checkpoint/resume**, unlike B1/B4. Their
    measurement is delegated entirely to an UNMODIFIED upstream
    `test_quant.py` subprocess, whose own `validate()` loop has no
    checkpoint hook -- adding one would require either modifying their
    file (violates this project's fairness rule) or fully reimplementing
    their quantization+eval pipeline ourselves (defeats the purpose of
    running their own code, and risks silently diverging from their
    exact protocol). A Colab disconnect mid-run restarts B2/B3 from
    scratch. Partial mitigation: B2/B3's own single `validate()` pass is
    structurally much shorter than B1's 3-r-value sweep or a
    hypothetical longer job, so the exposure window is smaller, not
    eliminated.
15. **Calibration image indices are reconstructed via a side computation,
    not observed from the real subprocess.** `common/calib_capture.py`
    replicates RepQ-ViT's/FQ-ViT's exact `seed()` -> `ImageFolder(train)`
    -> `DataLoader(shuffle=True)` call sequence (verified against their
    real source at the pinned commits) and reads the resulting
    `RandomSampler`'s own index sequence directly -- proven equivalent by
    inspecting PyTorch's own `RandomSampler.__iter__` source (confirmed:
    `generator=None` draws one seed from the global RNG state, which is
    identical in both the capture script and the real run since nothing
    else consumes the global RNG between `seed()` and the first
    DataLoader iteration in either code path), not executed side-by-side
    against a real run to CONFIRM the recovered indices are byte-for-byte
    what the real run used (no GPU in this environment) -- the first real
    run on either host is the actual confirmation.

## Checkpoint-identity policy + FP-parity gate (pre-run finalization) — new deviations

16. **RepQ-ViT's and FQ-ViT's own ViT-B checkpoint is NOT TANDEM's pinned
    tag.** Both repos' unmodified `--model vit_base` resolves to the
    checkpoint at `storage.googleapis.com/vit_models/augreg/
    B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-
    steps_20k-lr_0.01-res_224.npz` -- confirmed byte-identical to modern
    timm's `vit_base_patch16_224.augreg_in21k_ft_in1k` `url=` field (real
    source diff, not assumed) and to timm==0.4.12's own (pre-tag-system,
    single-checkpoint) `default_cfg` for `vit_base_patch16_224` (both
    pinned commits/timm versions predate timm's per-model tag system).
    Independently cross-confirmed by each repo's own README: RepQ-ViT
    lists "ViT-B (84.54)", FQ-ViT lists ViT-B Full Precision = 84.53 --
    both match `augreg_in21k_ft_in1k`'s real accuracy (84.534), a 0.57pp
    gap from TANDEM's pinned `augreg2_in21k_ft_in1k` (FP 85.108, P2.1 S8
    measured 85.10) -- not noise. Every DeiT model's own unmodified
    pretrained-loading DOES coincide with TANDEM's pinned `.fb_in1k` tags
    (verified: both repos load byte-identical `dl.fbaipublicfiles.com/
    deit/*` URLs). See `common/checkpoint_pin.py`'s module docstring for
    the full verification chain.
17. **Two-step protocol, per results.csv row.** Step A (reproduction):
    every method runs at its own documented settings on ITS OWN paper
    checkpoint, gated against that paper's published number where one
    exists (UNCHANGED behavior for every model except that it is now
    explicitly labeled `row_kind=reproduction` and carries the resolved
    `checkpoint_tag`). Step B (comparison): the SAME method settings
    re-run on TANDEM's pinned checkpoint -- for DeiT this is the identical
    checkpoint as Step A, so no second GPU pass happens (`row_kind=
    comparison` rows reuse Step A's measured numbers, noted explicitly);
    for ViT-B on B2/B3 this is a REAL second run, with a checkpoint
    override injected via the same sitecustomize.py PYTHONPATH mechanism
    already used for TF32 (never a modification of either repo's file on
    disk -- see item 18). Comparison rows always carry
    `gate_status=not_applicable` (no published number exists for a method
    run on a checkpoint neither paper evaluated) -- their correctness rests
    on the FP-parity gate (item 19), not the +-0.3% reproduction gate.
18. **Checkpoint override mechanism: downloaded HF-Hub weights, injected
    via monkeypatch, never a file edit.** `common/checkpoint_pin.py`
    downloads `pytorch_model.bin` for TANDEM's pinned ViT-B tag directly
    from `huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`
    (a plain, wrapper-free torch state dict -- confirmed via the real
    safetensors header on that repo: standard timm `VisionTransformer` key
    names, identical to old timm==0.4.12's own attribute names, confirmed
    against its real source) and, gated by `TANDEM_PIN_CHECKPOINT_OVERRIDE=1`
    (set ONLY for a deliberate Step-B call), monkeypatches
    `timm.create_model` (RepQ-ViT, via `build_model()`) or
    `models.utils.load_weights_from_npz` (FQ-ViT, via their vendored
    `vit_base_patch16_224()`) inside a `sitecustomize.py` on the
    subprocess's PYTHONPATH -- the SAME non-invasive mechanism
    `common/determinism.py` already uses for TF32 (extended with an
    `extra_sitecustomize_source` param so the two share one file, since
    Python's `site` module only auto-imports one module actually named
    `sitecustomize`). **This override's correctness has NOT been
    execution-verified** (no GPU/torch in this environment) beyond a
    static key-structure comparison (safetensors header vs. timm's real
    `VisionTransformer.__init__` attribute names) -- the FP-parity gate
    (item 19), which runs immediately after and before any method-specific
    number is trusted, is the actual empirical proof it worked.
19. **FP-parity gate is a NEW full-50k pass, not a smoke check.** Before
    any method's config sweep touches TANDEM's pinned checkpoint (Step A
    for DeiT, Step B for ViT-B), a plain/unquantized pass through that
    method's own model construction + data loading (B1/B4: skip the
    ToMe/RepQ steps entirely; B2: `build_model()` only, no
    `quant_model()`; B3: their own constructor's `quant=False` default,
    never reaching `model_quant()`) is evaluated on the full 50k set and
    gated at +-0.15pp against P2.1's own S8 measurement (72.166 / 79.848 /
    81.97 / 85.10). This is real, additional GPU time beyond what this
    runner previously spent (one extra full-50k pass per model, not per
    config) -- deliberate, since a smoke-subset check (already covered by
    `common/equivalence_gate.py`, a DIFFERENT question: cross-host
    reproducibility of a baseline's OWN numbers) would not reliably catch
    a harness-level preprocessing/eval-loop discrepancy on the full set.
20. **One cost ruler: ToMe's N_l is now a closed-form schedule, not only a
    live measurement.** `common/bops.tome_schedule_n_l(r)` implements
    `N_l = n_tokens_total - r*(l-1)` (r merged after EVERY block) --
    verified against ToMe's real `tome/utils.py` `parse_r` source: at the
    default `inflection=0` (never overridden anywhere in this project),
    `parse_r` returns a CONSTANT r at every block, i.e. exactly this
    formula. `bops_identity_check.py` unit-checks `tome_schedule_n_l(0)`
    against `C_id` exactly (pure Python, no GPU). B1/B4 now use this
    formula as the source of truth for cost and ASSERT it against the
    pre-existing live forward-hook measurement (`measure_token_trajectory`)
    -- a real run where the two disagree halts rather than silently
    reporting either number.
21. **DeiT-Tiny rows added to B2/B3 (spec item #5).** Both RepQ-ViT's and
    FQ-ViT's READMEs publish a DeiT-T row (RepQ-ViT: W6/A6 only, same
    "no W8/A8 located" gap as their other three models, see item 5; FQ-ViT:
    W8A8 "Ours" = 71.61). ToMe (B1) and the sequential composites (B4)
    remain DeiT-S only -- ToMe's README publishes no DeiT-Tiny row, and B4
    was scoped to DeiT-S only per the original spec; neither is extended
    here (per this item's own instruction: "skip any model a paper doesn't
    report rather than inventing a config").

## First real Colab run — two defects caught, both fixed and hardened against recurrence

22. **`common/fp_parity_gate.py`'s `_repq_fp_pass` called RepQ-ViT's real
    `build_model()` with the short CLI alias (e.g. `"deit_tiny"`) instead
    of the full timm name (`"deit_tiny_patch16_224"`) that `build_model()`
    actually expects.** `build_model()` itself does NOT do the
    short-to-full translation -- that only happens inside
    `test_quant.py`'s own `main()` (a function-local `model_zoo` dict),
    which the FP-parity gate deliberately never calls (it stops before
    quantization). The docstring already stated the correct call
    (`build_model(model_zoo[args.model])`); the code below it didn't
    actually do the lookup. Surfaced immediately and loudly on first real
    use (`RuntimeError: Unknown model (deit_tiny)`) -- never reached a
    reported number. Fixed by duplicating RepQ-ViT's real `model_zoo`
    dict (verified against their live source at the pinned commit) inside
    `_repq_fp_pass`. Re-traced the full Step-A and Step-B (checkpoint-
    override) call chains end-to-end for both RepQ-ViT and FQ-ViT after
    the fix -- `_fqvit_fp_pass` never had this bug (`str2model` maps the
    short alias directly to the constructor, no intermediate name to get
    wrong), and the checkpoint-override sitecustomize matching (item #18)
    is unaffected since it already keyed off the resolved full name.
23. **`common/colab_imagenet.py` named ImageFolder class directories with
    `ClassLabel.int2str(label)`, which returns a human-readable class
    DESCRIPTION (e.g. "tench, Tinca tinca"), not the WNID its own variable
    name (`wnid`) claimed.** `torchvision.datasets.ImageFolder` assigns
    class indices by alphabetically sorting subdirectory names; sorting
    human-readable descriptions produces a completely different ordering
    than sorting WNIDs, silently scrambling the class-index-to-model-
    output correspondence while every individual image still landed in
    its own (correctly-populated) class folder -- images right, labels
    wrong. This is NOT a narrow bug: every accuracy number ever produced
    through a Colab-materialized ImageNet copy (any baseline, any config)
    would have been silently invalid, not just the FP-parity gate that
    happened to surface it. Caught on first real end-to-end use: a plain
    pretrained DeiT-Tiny scored 0.47% top-1 (1000-class random chance is
    ~0.1%) through the FP-parity gate, rather than the ~72% every other
    verification in this project would predict. Root-caused via HF's own
    documentation ("the labels are indexed based on a sorted list of
    synset ids" -- i.e. the RAW integer label is already assigned by the
    same alphabetical-WNID-sort convention `ImageFolder` itself uses).
    Fixed: class directories are now named by the zero-padded raw integer
    label (`f"{label:04d}"`) directly, never `int2str()`, for both
    `materialize_val_split` and `materialize_train_calib_subset`.
    **Hardened against recurrence, not just fixed once:** (a)
    `common/colab_imagenet.py` now asserts every class directory matches
    `^\d{4}$` in [0000,0999] on EVERY call (including the idempotent
    "already materialized, skip" path, so a directory tree built by the
    old buggy code is caught on the very next run, not silently reused);
    (b) `common/imagenet.py` (the university-GPU / local path, never
    affected by this specific bug, but auditing the same class of risk)
    now asserts exactly 1000 class directories are present; (c)
    `common/fp_parity_gate.py` and `common/eval_utils.py` both gained an
    early-abort check (first 3 batches; abort below 3% top-1) so a
    catastrophically broken harness of ANY kind -- not just this one --
    fails in seconds on both the Colab (B2/B3 subprocess) and in-process
    (B1/B4) paths, instead of costing a full 50k-image pass (~6.5 minutes
    here) to discover.
24. **RepQ-ViT's own `--num-workers 16` default (their documented default,
    inherited unchanged by B2's subprocess call) OOM-killed DeiT-Base's
    w8a8 config on Colab's 2-CPU runtime, after DeiT-Tiny and DeiT-Small
    had both fully completed and passed their FP-parity + reproduction
    gates.** Evidence: `RuntimeError: RepQ-ViT test_quant.py failed (exit
    -9)`, with PyTorch's own `UserWarning` in the captured log noting 16
    DataLoader workers were requested against a suggested max of 2 --
    exit -9 is a SIGKILL, not a Python exception, consistent with the OS
    OOM-killer terminating the process once DeiT-Base's larger quantizer-
    wrapper memory footprint combined with 16 real worker PROCESSES (not
    threads -- each with its own copy of the dataset index and transform
    pipeline) exceeded available RAM. This is a throughput/memory
    ENGINEERING knob, not a scientific or fairness-rule setting -- worker
    count never changes which images are loaded, their order
    (`shuffle=False` throughout), or any other protocol detail that
    affects the measured number -- so adapting it per-host is not a
    deviation from either baseline's published protocol. Fixed by capping
    every `--num-workers` default to `min(<method's own documented
    default>, os.cpu_count() or <small fallback>)`: `B1_tome`/
    `B4_sequential_composites` via `common/imagenet.py`'s
    `DEFAULT_NUM_WORKERS`; `B2_repq_vit`/`B3_fq_vit`'s `run.py` via a
    module-level `NUM_WORKERS` constant now passed through explicitly as
    `--num-workers` to the upstream subprocess (previously not passed at
    all, silently falling through to the upstream script's own hardcoded
    default); `common/fp_parity_gate.py`'s own `_cli()` (a separate cheap
    side-pass, same risk) via the same `min(16, os.cpu_count() or 2)`
    pattern. All remain overridable via an explicit `--num-workers` flag.
    **Also hardened the sweep itself against this class of interruption,
    not just its cause:** a crash partway through a bit-sweep or FP-parity
    gate previously meant re-running the entire baseline from scratch,
    re-doing (and re-paying GPU minutes for) every already-completed,
    already-recorded config. `common/results_io.find_existing_row()` (new)
    lets `B2_repq_vit`/`B3_fq_vit`'s `run.py` check `results.csv` for a
    matching `(method, model, config, row_kind, checkpoint_tag)` row
    before running that config, skipping it if already present -- unit-
    verified (exact-match found; a different `config` or `row_kind` on an
    otherwise-identical key correctly does NOT match, so e.g. DeiT-Base's
    w6a6 row is never mistaken for its w8a8 row). FP-parity gate passes
    are separately cached to small JSON files under `_fp_parity_cache/`
    (gitignored, see `.gitignore`'s `_*/` pattern), written only after a
    genuine pass, so a cache hit is always a real prior success, not an
    assumption.
25. **Colab/university-GPU session-length limits (a real availability
    constraint, not a defect) motivated splitting each baseline's single
    "run everything" Colab cell into several smaller, independently
    resumable cells -- one per natural sub-unit of work.** B1/B4's `run.py`
    gained `--only-r`/`--only-config` (B1: one of r in {8,13,16}; B4: one
    of the two composites); B2/B3's gained `--only-model` (one of the four
    models) -- chosen because that is exactly the granularity item #24's
    real DeiT-Base OOM crash happened at (two models already succeeded,
    the third crashed). Every flag defaults to "run everything", so a
    plain `python run.py` on the university GPU is completely unchanged.
    Each `*_colab.ipynb`'s single run cell is now several cells (one per
    r-value/model/composite), each with its own markdown explaining what
    it does and, where relevant, which published number it's gated
    against -- so a session limit or crash on one cell costs only that
    cell's work, not the whole baseline. This exposed and closed a real
    latent gap along the way: B1/B4 (unlike B2/B3) had NO results.csv-
    level "already recorded, skip" check -- only a per-batch checkpoint
    for an INTERRUPTED pass (`evaluate_top1_resumable`). Restarting
    `main()` after a full process exit (which any Colab-disconnect-driven
    notebook restart is) would re-enter the r-value/composite loop from
    the top and re-append an already-recorded row as a duplicate. Fixed by
    giving B1/B4 the same `common.results_io.find_existing_row` per-
    config resume check B2/B3 already had (item #24), and by caching their
    in-process FP-parity gate pass to `_fp_parity_cache/` (mirroring B2/
    B3's subprocess-gate caching) so splitting into multiple cells doesn't
    force repaying that full-50k pass on every cell. Each notebook's final
    "Results" cell is now also a completeness check: it asserts the exact
    expected row count for that baseline (6 for B1, 16 for B2, 8 for B3, 4
    for B4 -- derived from each baseline's own model/config/row_kind
    count, not guessed) and fails loudly, naming which cell to re-run, if
    any expected row is missing -- so an incomplete run can never be
    mistaken for a complete one before results reach `compare_pareto.py`
    or manual review. The 5 Colab notebooks remain separate, standalone
    files (not merged into one) specifically because Colab and university-
    GPU sessions both have real time limits -- one notebook per baseline
    keeps each one's total runtime, and the blast radius of any one
    session ending early, bounded and independent of the other baselines.
26. **P2.2 post-conversion self-test cell (specified, 10 hard-assert
    checks), added as the first executable cell after bootstrap in every
    `*_colab.ipynb`, before any real run cell.** New module
    `common/notebook_self_test.py`. Most of the 10 items were already
    implemented elsewhere in `common/` from this project's own prior audit
    trail -- this mostly SURFACES those existing guarantees into one
    printed PASS/FAIL banner per notebook, reusing the real functions
    rather than re-implementing them: wrapper purity reuses
    `verify_colab_thin_wrappers.check_file`; upstream pinning reuses
    `repo_utils.ensure_repo`; the numeric-environment check reuses
    `determinism.disable_tf32`; cost-ruler checks reuse
    `bops_identity_check`'s existing functions verbatim; schema/resume
    reuse `results_io.RESULTS_COLUMNS`/`find_existing_row`; the
    equivalence-fingerprint check reuses `equivalence_gate.run_gate`
    unmodified. Genuinely new: TANDEM-repo-HEAD recording, the secrets
    regex scan, the val-manifest hash, and a REAL interrupt-then-resume
    simulation (B1/B4 only -- a 512-image slice is evaluated once
    uninterrupted, then again with a simulated kill after 5 batches
    followed by a real resume via `eval_utils.evaluate_top1_resumable`,
    asserting the two runs' correct-counts match EXACTLY and that resuming
    against a wrong config_hash correctly starts fresh rather than reusing
    stale state). A cell failure raises, halting "Run All" before any
    `--only-*` run cell executes -- verified this halts wrapper-purity and
    secrets violations on planted test cases (a fake `hf_`/`github_pat_`/
    `ghp_` literal, a forbidden `torch.no_grad()` pattern, a missing
    results.csv column), not just trivially passing on clean input.

    **Three places where the original spec's literal wording doesn't fit this
    project's actual, already-audited design -- resolved explicitly, not
    silently weakened (confirmed with the user before implementing):**
    (a) the TANDEM repo's own commit is RECORDED (`git rev-parse HEAD`,
    asserted well-formed, printed, saved) rather than hard-pinned to one
    frozen SHA -- unlike ToMe/RepQ-ViT/FQ-ViT (frozen third-party
    research code), `baselines/` is our own actively-developed code; a
    hard pin would need manual upkeep on every push and would falsely
    fail the moment any future improvement is pushed; (b) the schema
    check asserts our real 19-column `RESULTS_COLUMNS` is a SUPERSET of
    the spec's 14 named columns (case-insensitive), not exact
    equality -- the 5 extra columns (`gate_status`, `gate_diff_pct_points`,
    `calib_images`, `published_top1_pct`, `cuda_version`) are load-bearing
    in this project's own gates; deleting them to match a shorter list
    would be a regression; (c) the data-determinism check does not
    literally diff against `experiments/phase2_1_fleet/P2_tandem_pipeline.py` (already-
    validated, thesis-relevant, out of this session's scope to touch) --
    it hashes baselines' own materialized val manifest and stores it as
    baselines' own canonical fingerprint (same first-run-defines-canonical
    pattern as `equivalence_gate.py`, sharing its `equivalence_fingerprint.json`
    file via a new `_manifest_hashes` key). **Caught during this session's
    own audit, before it could mislead anyone:** the module's first draft
    docstring claimed this catches drift "Colab session-to-session, Colab
    vs university GPU" -- the second half is false and was corrected.
    `notebook_self_test.py` is only ever invoked from the Colab notebooks,
    never from the university-GPU `run.py` path, and the two hosts
    materialize ImageNet with DIFFERENT class-directory naming conventions
    (`common/colab_imagenet.py`: zero-padded integer label, per item #23's
    fix; `common/imagenet.py`: whatever real WNID names the user's local
    copy already has) -- comparing hashes across those two naming
    conventions would produce a FALSE mismatch for the identical
    underlying image set, not a real one. The check therefore only
    validates drift across Colab sessions (docstring and comments
    corrected to say so explicitly, not silently). Literal identity to
    P2.1's own S8 measurement is asserted INDIRECTLY either way, via the
    FP-parity gate's existing 0.15pp accuracy check against P2.1's own S8
    number -- a scrambled/wrong manifest would fail that gate regardless
    of host (see item #23, the class-directory-scrambling bug that
    manifested exactly this way).
27. **Self-audit of the new `notebook_self_test.py` module (item #26)
    before any real Colab/GPU run, since none of its checks had been
    exercised end-to-end yet.** Found and fixed two issues, neither of
    which had shipped to a real run: (a) the "Colab vs university GPU"
    overclaim above; (b) `check_gate_ordering` (item 10) asserted checks
    6 and 9 both passed WITHOUT checking whether they were actually in
    scope for that invocation -- under a partial `--checks-only` selection
    that includes 10 but excludes 6 or 9 (not exercised by any current
    notebook wiring, which either runs all 10 or B5's fixed 1,4,7, but a
    real latent inconsistency regardless), it would have reported a false
    "check did not pass" rather than correctly treating an out-of-scope
    check as not applicable. Fixed by scoping the ordering check to only
    the (6, 9) subset actually present in `checks_to_run`. Verified via
    direct execution (no GPU needed for this part): all 5 torch-free
    checks (1, 4, 5, 6, 7) against this real repo and `results.csv`
    schema; negative tests proving checks 1 and 4 actually catch planted
    violations (a fake `hf_`/`github_pat_`/`ghp_` token literal, a
    forbidden `torch.no_grad()` pattern) rather than trivially passing; a
    real `ensure_repo` clone-and-pin of RepQ-ViT at its exact pinned
    commit (item 2); the `_worker_numeric_env` subprocess-dispatch
    mechanism (item 3) confirmed to correctly resolve the `common`
    package via `cwd` and fail with a clear, unmasked error (missing
    `torch`, this environment has none) rather than a confusing
    `ModuleNotFoundError: common`. Checks 3's real TF32 assertion, 8's
    B1/B4 resume simulation, and 9's equivalence-fingerprint smoke eval
    still cannot be verified from this environment (no GPU) -- these need
    a real Colab/university-GPU run.
28. **All 5 notebooks invoked the new self-test as `python -m
    common.notebook_self_test ...`, which fails with `ModuleNotFoundError:
    No module named 'common'` from the notebooks' actual cwd.** Surfaced
    on the FIRST real Colab run of B2, immediately at the self-test cell.
    Root cause: `python -m package.module` resolves the package via the
    CURRENT WORKING DIRECTORY, which needs to be `baselines/` for
    `common` to be found -- but every notebook's clone cell does `%cd
    /content/tandem` (the REPO ROOT, one level above `baselines/`), and
    every other invocation in these notebooks (`python
    baselines/bN_xxx/run.py ...`) is a direct SCRIPT PATH from that same
    root, not a `-m` module invocation, so this mismatch never surfaced
    anywhere else. My own verification before this session's earlier
    response used `cd baselines && python3 -m common.notebook_self_test
    --help`, which accidentally matched the `-m` mechanism's cwd
    requirement while not matching the real notebooks' actual cwd --
    tested from the wrong directory, so the bug passed unnoticed. Fixed
    by changing all 5 notebooks' invocations from `python -m
    common.notebook_self_test ...` to `python baselines/common/
    notebook_self_test.py ...` (a direct script path, matching every
    other invocation in these notebooks) -- `notebook_self_test.py`
    already does its own `sys.path.insert(0, .../baselines)` at import
    time (needed for ITS internal `from common import ...` lines), so
    this resolves correctly regardless of invocation style. The internal
    worker-dispatch subprocess call (item 3's numeric-environment check,
    `[python_exe, "-m", "common.notebook_self_test",
    "_worker_numeric_env"]`) was NOT affected -- it already passed an
    explicit `cwd=str(BASELINES_ROOT)`, unrelated to the outer script's
    own invocation style. Re-verified this time from the REAL cwd
    (repo root, `cd "tandem" && python3 baselines/common/
    notebook_self_test.py ...`, not `baselines/`) for checks 1/2/4/7 and
    check 3's worker-dispatch path (fails only on missing torch in this
    sandbox, the correct/expected failure here -- not a masked
    `ModuleNotFoundError: common`).

    B5 (`compare_pareto.py`) does no GPU/upstream-repo/calibration/cost-
    model work, so it runs a lighter subset (`--checks-only 1,4,7`:
    wrapper purity, secrets, schema) via the same module's new
    `--checks-only` filter, rather than forcing 7 inapplicable checks.
    Could not execute the GPU-dependent checks (numeric environment,
    B1/B4's resume simulation, the equivalence-fingerprint smoke eval)
    from this environment (no GPU here) -- these need a real Colab/
    university-GPU run to confirm end-to-end, same limitation as every
    other GPU-dependent change in this file.
29. **`common/equivalence_gate.py`'s smoke-equivalence check (item 9)
    invokes each baseline's real `run.py` on a 256-image subset --
    `run.py`'s own internal FP-parity gate (0.15pp tolerance) HALTS on
    that subset every time, for any DeiT model, regardless of
    correctness.** Surfaced on the first real Colab run of item 9 (B2,
    DeiT-Tiny Step A): `harness=82.8125% P2.1_S8=72.1660% diff=10.6465pp
    (tol=0.15pp) -> FAIL`. Root cause, confirmed by the numbers, not
    guessed: (a) the smoke subset is the FIRST 256 images in sorted
    class order (`equivalence_gate._sorted_samples`) -- with 50
    images/class, that's only ~5 of ImageNet-1k's 1000 classes, not a
    representative sample, so a plain model's accuracy on just those
    classes can differ sharply from its true 1000-class accuracy; (b)
    more fundamentally, EVEN a representative 256-image sample carries
    ~2.8pp of sampling standard error (binomial SE at n=256, p~0.72) --
    reliably hitting 0.15pp would need roughly 90,000+ images, more than
    the entire 50k val set. No subset size or sampling strategy could
    make this gate pass at smoke scale; the tolerance is calibrated
    entirely for full-50k statistics. `equivalence_gate.py` (pre-existing
    code, not written this session) and the FP-parity gate (also
    pre-existing) had apparently never actually been run together before
    this session's self-test module became the first code path to wire
    `run_gate()` into a real, executed flow -- an architectural
    incompatibility between two independently-correct pieces of code,
    not a defect in either one alone. Fixed via the SAME env-var-override
    precedent `TANDEM_RESULTS_CSV_OVERRIDE` already established for
    exactly this situation (smoke run needs to override normal `run.py`
    behavior without a code fork): `equivalence_gate._run_smoke` now also
    sets `TANDEM_SMOKE_RUN=1` on the subprocess env (renamed from an
    initial `TANDEM_SKIP_FP_PARITY_GATE` once item #30 below found this
    same flag needed to gate a SECOND check too); every baseline's
    FP-parity-gate call site (B1/B4 in-process, B2/B3's
    `run_fp_parity_gate` wrapper) checks this and skips (NOT cached --
    a smoke-scale skip must never satisfy a later real run's own cache
    check) before doing any work. Does not weaken any real guarantee:
    the actual 0.15pp gate still runs, unmodified, on every genuine
    full-50k pass (as it already has, successfully, in this session's
    earlier real B2 runs on DeiT-Tiny/Small) -- the smoke run's own
    purpose (cross-host correct-count reproducibility on a fixed subset)
    is a different, unaffected check. Verified by actually exercising
    both branches of the new code (not just reading the diff): the skip
    path returns immediately even with deliberately bogus arguments
    (proving it never reaches real logic when set), and the normal path
    (env var unset) proceeds all the way to attempting the real
    subprocess call (proving the skip check doesn't accidentally fire on
    a genuine run).
30. **Two more issues found in the same class, during a final proactive
    audit pass BEFORE any real re-run (user asked to "check all design
    bugs... one last time" before their evening session) -- neither
    triggered yet, both fixed pre-emptively:**
    (a) **The reproduction gate (0.3pp tolerance, `common/repro_gate.py`)
    has the EXACT SAME small-sample incompatibility as item #29's
    FP-parity gate** -- it also runs inside `run.py`'s normal flow for
    every `row_kind="reproduction"` config (B1, B2, B3 -- B4 has no
    reproduction gate at all, its `gate_status` is a hardcoded
    `"not_applicable"`, confirmed by re-reading its composite loop), and
    would have HALTED the smoke run at the very next step after item
    #29's fix let it past the FP-parity gate. Same numbers apply (~2.8pp
    standard error at n=256 vs a 0.3pp tolerance) -- confirmed, not
    guessed, by re-deriving the tolerance from `common/repro_gate.py`'s
    own `GATE_TOLERANCE_PCT = 0.3`. Fixed the same way: `TANDEM_SMOKE_RUN`
    (renamed from the narrower `TANDEM_SKIP_FP_PARITY_GATE` now that it
    gates two different checks) makes the reproduction-gate assertion in
    B1/B2/B3 non-fatal under a smoke run -- still computed and printed
    (visible), just not asserted. Verified: `check_gate()` still correctly
    returns `"fail"` for a real mismatch (72.166% vs 82.8125%, the exact
    numbers from the user's failed run) when called directly, confirming
    the real gate's own logic is completely unchanged -- only the smoke
    run's bypass is new.
    (b) **B1/B4's per-batch eval checkpoint (`_checkpoints/<config>.json`)
    is keyed by a `config_hash` that does NOT include dataset identity or
    size** (confirmed by re-reading each `config_hash(...)` call site --
    neither passes `imagenet_dir` or an image count). Since
    `equivalence_gate.py`'s smoke check runs the SAME `run.py`, for the
    SAME r-value/composite, on a 256-image subset that is BY CONSTRUCTION
    a symlinked prefix of the real 50k set (`build_smoke_subset`) -- a
    smoke run's checkpoint and a real run's checkpoint for the same
    config would collide under the identical hash, and a real run could
    silently "resume" from a smoke run's partial or complete 256-image
    state. Under TODAY's specific defaults (uniform batch size, identical
    sort order) this would likely still land on a numerically correct
    final count by coincidence -- but that's an accident of several
    fragile, undocumented assumptions holding simultaneously, not a
    guarantee, and this project does not accept "probably fine" for
    anything that affects a reported number. Fixed with the same
    isolation pattern `TANDEM_RESULTS_CSV_OVERRIDE` already established:
    `equivalence_gate._run_smoke` now also sets
    `TANDEM_CHECKPOINT_DIR_OVERRIDE` to a smoke-only directory
    (`_smoke_checkpoints/<baseline>/`), and B1/B4's checkpoint-path
    construction reads it (falling back to the original `_checkpoints/`
    path when unset, i.e. every real run is completely unaffected).
    Verified the Path-construction logic directly for both the override
    and no-override cases. Neither (a) nor (b) had actually fired yet in
    any real run (the user's B2 run halted at item #29's FP-parity gate
    before reaching either) -- found by re-tracing the full smoke-run
    code path end-to-end rather than waiting for a second real failure.
31. **B1 setting-mismatch correction: our off-the-shelf r=13 run
    was being gated against ToMe's TRAINED-checkpoint number, not a real
    reproduction target.** Found after a real Colab run: once B1's
    preprocessing bug (see `common/imagenet.py`'s module docstring, the
    bicubic/crop_pct=0.9 fix) was corrected, the r=13 gate
    still failed (measured 78.94% vs "published" 79.36%, diff 0.42pp,
    tolerance 0.3pp) even though the same fix made FP-parity match ToMe's
    own stated 79.8% baseline almost exactly (79.826%, diff 0.026pp).
    Root cause (verified by fetching ToMe's README at the pinned commit
    and the paper's appendix directly, not assumed): ToMe publishes DeiT-S
    numbers for two different things that happen to share the r=13 label
    -- an OFF-THE-SHELF setting (our run: a pretrained checkpoint, ToMe
    applied post-hoc at eval time only) and a TRAINED-WITH-TOME setting
    (their run: a DIFFERENT checkpoint, `deit_S_r13.pth`, fine-tuned FROM
    SCRATCH with merging active throughout training). The README's DeiT
    table is the trained setting only; no off-the-shelf DeiT-S reference
    exists anywhere in ToMe's README or paper (see corrected item #3).
    79.36% was never a valid gate target for our off-the-shelf run --
    comparing across settings, not a reproduction failure.

    **Correction applied**, `b1_tome/published.json` restructured
    (r13 now `{"off_the_shelf": {...gate:false...}, "trained_with_tome":
    {"top1_pct": 79.36, "gate": true, "checkpoint_url": ...}}`) and
    `results.csv` gained a `setting` column (`off_the_shelf` |
    `trained_with_tome` | `""` for every other baseline) --
    `compare_pareto.py` and any TANDEM-vs-baseline plot must read ONLY
    `setting=off_the_shelf` (or `""`) rows; a `trained_with_tome` row is a
    reference point, never a fair training-free comparison (see
    `common/results_io.py`'s schema comment). Two BINDING gates now stand
    in place of the old (wrong) r=13 gate, both added to `b1_tome/run.py`:
    (a) `r0_baseline` -- the plain/unpatched off-the-shelf checkpoint
    (already evaluated by the FP-parity gate), additionally gated against
    ToMe's own stated 79.8% baseline, ±0.3pp (not just P2.1's S8 79.848%,
    ±0.15pp) -- free, reuses the FP-parity pass, no second GPU pass;
    (b) `r13`/`trained_with_tome` -- downloads ToMe's own `deit_S_r13.pth`
    (`_download_tome_checkpoint`, same idempotent tmp-file-then-replace
    pattern as `common/checkpoint_pin.py`'s HF-Hub downloader), loads it
    per the README's own instructions ("instantiate a DeiT timm model,
    patch it with the timm patch (`prop_attn=True`), and use ImageNet mean
    and variance for data loading"), evaluates on the full 50k val set,
    gates against 79.36% ±0.3pp. Both HALT on failure, same as every other
    binding gate in this project. Neither has been run on a real GPU from
    this environment (no GPU available) -- the checkpoint's exact
    `torch.load` shape (assumed `{"model": state_dict, ...}`, the standard
    facebookresearch/deit training-script convention, since the README
    says these were "trained ... using the official implementation") is
    the one part of this correction NOT independently confirmed byte-for-
    byte; `state_dict.keys()`/`strict=True` will surface a shape mismatch
    loudly on the first real run if this assumption is wrong.

    Also found during the paper-appendix search (informational, NOT wired
    into any gate -- out of this correction's scope): Table 11 gives
    TRAINED (not off-the-shelf) numbers for r=8 (79.68%) and r=16
    (79.13%) too, and a slightly different trained r=13 number (79.42%,
    vs the README's 79.36% for the actual downloadable checkpoint --
    likely ordinary run-to-run training variance between the paper's
    official table and the specific checkpoint file uploaded later). Not
    gated against since no downloadable checkpoint backs the r=8/r=16
    paper numbers the way `deit_S_r13.pth` does for r=13.
32. **BOPs high-r cost-trajectory cap, verified to change zero
    actual numbers.** `common/bops.py`'s `tome_schedule_n_l` previously
    used a purely linear closed form (`N_l = n_tokens_total - r*l`,
    floored at 1) that never accounted for ToMe's real per-block merge
    cap: `bipartite_soft_matching` can never merge more than
    `floor((N_l - protected) / 2)` pairs (protected=1, the CLS token,
    never a merge candidate). Implemented the real, capped recurrence
    instead. **Verified this changes NOTHING for this project's actual
    usage** (`n_blocks=12`, `n_tokens_total=197`, r in {8,13,16} --
    B1/B4's only shape): computed both the old linear and new capped
    trajectories directly (pure arithmetic, no GPU) and confirmed
    byte-identical arrays for all three r values -- the cap only would
    bind at the terminal block (block 12), whose own merge OUTPUT feeds no
    13th block and therefore never appears in the entering-token-count
    array that cost is actually computed from. This is also independently
    confirmed by `b1_tome/run.py`'s own pre-existing live forward-hook
    cross-check (`assert n_l_formula == n_l_per_block`), which had already
    passed on real 50k GPU runs at r=8/13/16 before this fix even existed.
    No `bops_ours` number in `results.csv` changes as a result of this fix
    -- it is a correctness/future-proofing improvement (a future r/block-
    count/token-count combination COULD hit the cap early and silently
    understate cost otherwise), not a retroactive correction. Regression-
    tested: `common/bops_identity_check.py` gained a golden-value check
    (r=8/13/16 arrays pinned exactly) and a synthetic small-shape check
    proving the cap genuinely engages (r=50/n_blocks=3/n_tokens=20 would
    give the old formula's physically-meaningless `[20, 1, 1]`; the capped
    formula correctly gives `[20, 11, 6]`).

    **The cap is non-binding for every r this project actually runs, and
    the closest call is explicit**: at r=16 (the largest r used), the
    MINIMUM entering-token count across all 12 blocks is 21 (block 12,
    `[197, 181, 165, 149, 133, 117, 101, 85, 69, 53, 37, 21]`) -- the cap
    only would have started clipping below `floor((N-1)/2) < 16`, i.e.
    below N=33, and 21 is already the LAST block's entering count with no
    13th block to feed into, so the cap never actually fires for r in
    {8,13,16} in this project's 12-block/197-token shape. Confirmed by the
    real 50k run too: DeiT-S r=16 measured `bops_ours=6.277324e+11`,
    matching the pre-cap-fix formula exactly (see `results.csv`'s r=16 row).

    **Cap formula, corrected and stated explicitly for the record
    (record patch, item 1).** The per-block merge cap is
    **`min(r, floor((N-1)/2))`** -- the requested merge count `r`, capped
    by the maximum number of pairs `bipartite_soft_matching` can physically
    form once the CLS token is excluded from pairing (`N-1` mergeable
    tokens, `floor((N-1)/2)` pairs). This is exactly what
    `common/bops.py`'s `tome_schedule_n_l` already implements
    (`actual_r = min(r, max((n - protected) // 2, 0))`, `protected=1`) --
    the code was never wrong. An earlier spot-check had proposed
    the CLS-unaware variant `floor(N/2)` (no `-1`, i.e. treating the CLS
    token as mergeable); that variant is the wrong one and is superseded by
    this entry. Validated against the live forward-hook cross-check
    already described above (`assert n_l_formula == n_l_per_block`,
    passing on real 50k GPU runs at r=8/13/16) -- no re-run needed, this
    is a documentation-only correction.
33. **Measured fine-tuning recovery at r=13, reported finding (not a gate),
    from the real B1 50k run.** Now that BOTH settings are
    measured on the same r=13 merge ratio (see item #31's correction):
    off-the-shelf (ToMe applied post-hoc, no training) measured 78.938%;
    trained_with_tome (ToMe's own checkpoint, fine-tuned WITH merging
    active throughout training) measured 79.230%. **Fine-tuning-with-
    merging recovers 0.292 percentage points** (79.230 - 78.938) relative
    to applying the identical r=13 merge ratio post-hoc to an off-the-shelf
    checkpoint, at the SAME token-reduction level. Reported as a finding,
    not gated against anything -- both numbers already passed their own
    respective gates independently (off-the-shelf r=13 is ungated, no
    published off-the-shelf reference exists, item #31; trained_with_tome
    passed its own gate, measured 79.230% vs published 79.36%, diff
    0.130pp, well within 0.3pp tolerance). Worth citing together in the
    thesis write-up as direct evidence of the training-free vs
    trained-with-method accuracy gap this project's whole B1 correction
    was about -- not just a bug fix, a real, small, measured number.
34. **B3 (FQ-ViT) DeiT-Tiny W8A8 real reproduction gap, ruled
    record-not-halt.** Real full-50k run: measured=71.042%,
    published=71.61% (FQ-ViT README, "Ours" row, 8/8/8, re-verified
    verbatim against their real README table, not a transcription error),
    diff=0.568pp, exceeding the 0.3pp tolerance -- the assertion halted
    the runner as designed, then was investigated the same way as item
    #31's B1 gap before any decision was made:
    - **FP-parity passed at 0.03pp** (72.196% vs P2.1's own S8 72.166%)
      on the SAME checkpoint and the SAME preprocessing the quantized run
      also uses -- rules out a harness-level bug (wrong checkpoint, wrong
      preprocessing, wrong labels).
    - **Calibration defaults independently confirmed to match FQ-ViT's
      own real source**: fetched their actual `test_quant.py` at the
      pinned commit -- `--calib-batchsize` defaults to 100, `--calib-iter`
      to 10, exactly what `b3_fq_vit/run.py`'s `CALIB_BATCHSIZE`/
      `CALIB_ITER` constants assume (neither is passed explicitly on the
      command line, relying correctly on their own defaults).
    - **Seeding mechanism already verified equivalent** (their `seed()`
      call, `DataLoader(shuffle=True)`'s `RandomSampler` behavior) against
      `common/calib_capture.py`'s own replication, built from their real
      source in an earlier session.
    - **The batch-by-batch trace shows normal class-to-class drift**
      (cumulative accuracy sliding from 77.7% to 71.0% as later classes
      enter, no sudden cliff or zero-accuracy stretch) -- not the
      signature of an indexing/label bug (contrast with the real
      class-scrambling bug documented in `common/colab_imagenet.py`'s own
      module docstring, which produced near-random ~0.1-0.5% accuracy).
    - DeiT-Tiny itself is not broken in this harness: the SAME model,
      SAME checkpoint, under a DIFFERENT PTQ method (B2/RepQ-ViT, W6A6)
      reproduced to 0.228pp, well within tolerance.

    No further root cause identified from code alone -- most likely a
    genuine small hardware/numerical PTQ reproduction gap (DeiT-Tiny is
    the smallest, least-redundant model this project runs, and min-max
    PTQ calibration is known to be sensitive to numerical differences in
    the forward pass used to compute activation ranges, which can differ
    slightly across GPU generations even under identical seeds -- this
    project's Tesla T4 is not necessarily what FQ-ViT's authors
    benchmarked on). **Ruling: record-not-halt** -- the 0.3pp
    tolerance itself is UNCHANGED (still asserted for every other config,
    still the same uniform bar every baseline is held to); only the
    FAILURE MODE changed, from a hard halt to recording `gate_status=fail`
    with the real measured number and continuing to the next model. This
    is the same category of decision as item #31, reached the same way:
    harness verified clean first, judgment call made explicitly and
    recorded here, never silently absorbed into the tolerance itself.

35. **[SUPERSEDES item #34's "no further root cause identified"] All four
    B3 Step-A `gate_status=fail` rows (DeiT-T/S/B, ViT-B) have a decisive,
    source-confirmed explanation: `--lis` hardcodes attention to 4-bit,
    not 8-bit -- our runs measure FQ-ViT's own published "8/8/**4**" row,
    not the "8/8/8" row `published.json` compares against.** (Found
    while investigating the ViT-B Step-B FP-parity failure per a
    documented ruling -- see item #36 below for that separate, still-open
    investigation.)

    Evidence, in order of discovery:
    - **GitHub issue search** (directed) on the pinned FQ-ViT repo
      turned up megvii-research/FQ-ViT#16, "Reproducing 8/8/8 for
      ViT-Base" (2022-06-03). An independent user ran the EXACT command
      this project's `run_test_quant()` runs (`--quant --ptf --lis
      --quant-method minmax`) and reported a deficit of almost identical
      shape to ours: ViT-B 82.540 vs published 83.31 (diff 0.77pp),
      DeiT-Small 78.370 vs published 79.17 (diff 0.80pp). The repo author
      (linyang-zhh) replied: *"We reported the 8/8/8 with Uniform
      quantization on Self-Attention, instead of Log2 ... you will need to
      set `BIT_TYPE_S = BIT_TYPE_DICT['uint8']`"* -- i.e. their published
      "Ours" 8/8/8 row was NOT measured with the flags their own README's
      "Run" section shows as the example command; getting true 8-bit
      attention requires an undocumented, source-level `config.py` edit,
      not exposed via any `test_quant.py` CLI flag.
    - **Confirmed directly against `config.py`'s real source at our pinned
      commit** (`5daf5915c715dc21d6afcd64e661652030e640ab`), not just the
      issue thread's say-so:
      ```python
      if lis:
          self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']   # attention forced to 4-bit
      else:
          self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']   # 8-bit, but LIS (their own
                                                       # contribution) then disabled
      ```
      `test_quant.py` has no CLI flag for `BIT_TYPE_S` at all (verified
      against its full argparse block) -- with `--lis` passed and
      `config.py` unmodified (this project has never touched it, per the
      fairness rule -- `run_test_quant()` runs `test_quant.py` completely
      unmodified), attention quantization is STRUCTURALLY 4-bit, not 8-bit,
      every single time. This is not a bug in their code or ours; it is
      what `--lis` does by design (their own table footnote: "Ours 8/8/4"
      is the row obtained by combining PTF+LIS with 8/8/8's OTHER two axes
      -- LIS's whole point, per the README's own Introduction section, is
      accurate quantization at "extreme lower bit-width (e.g. 4-bit)" for
      the attention map specifically).
    - **The numerical match to the "8/8/4" row, not "8/8/8", is exact
      within noise** for all four real Step-A measurements this project
      has:

      | Model      | Measured | Published 8/8/8 (what we gated against) | diff (8/8/8) | Published 8/8/**4** | diff (8/8/4) |
      |------------|----------|-------------------------------------------|---------------|----------------------|---------------|
      | DeiT-Tiny  | 71.042   | 71.61                                      | 0.568pp       | 71.07                | **0.028pp**   |
      | DeiT-Small | 78.454   | 79.17                                       | 0.716pp       | 78.40                | **0.054pp**   |
      | DeiT-Base  | 80.894   | 81.20                                       | 0.306pp       | 80.85                | **0.044pp**   |
      | ViT-B (Step A, augreg-v1) | 82.630 | 83.31                        | 0.680pp       | 82.68                | **0.050pp**   |

      Every model lands within 0.06pp of the "8/8/4" reference -- an order
      of magnitude tighter than the 0.3pp tolerance, and far tighter than
      the "8/8/8" comparison ever got. This retroactively corrects item
      #34's conclusion ("most likely a genuine small hardware/numerical PTQ
      reproduction gap") -- it is not hardware noise, it is a comparison
      against the wrong published row, now identified precisely.

    **Not fixed in code -- this is a methodology fork, not a bug fix, and
    is left for an explicit decision** (raised back to the user,
    same as items #26/#31/#34's judgment calls): either (a) point
    `published.json`'s `w8a8` comparison at FQ-ViT's own "8/8/4" row
    instead of "8/8/8" (honest about what `--lis`, run per their README's
    own example command, actually measures), or (b) add a config.py-level
    override (same non-invasive sitecustomize-injection technique already
    used for the Step-B checkpoint override in `common/checkpoint_pin.py`)
    to force `BIT_TYPE_S=uint8` and genuinely measure their true 8/8/8
    row. Note this does NOT affect `compare_pareto.py`/B5's own numbers:
    that script only reads `row_kind="comparison"` (Step B) rows and never
    references `published_top1_pct` in its comparison math (verified this
    session, re-reading `compare_pareto.py` in full) -- the mislabel is
    confined to the Step-A reproduction gate's own honesty about what it
    reproduced, not to any downstream Pareto comparison number.

    **Ruling, recorded for a self-contained record --
    no row values change:** this fork affects only the four FQ-ViT
    *reproduction* rows' gate interpretation. Comparison rows (which
    `compare_pareto`/B5 consume) use our own measured numbers and are
    unaffected. DeiT Step-A gaps are reported as documented reproduction
    deviations under our pinned protocol; published values are never
    substituted into comparison outputs. B3 (FQ-ViT) is accepted on this
    basis -- no re-evaluation of any model required.

36. **ViT-B Step-B FP-parity gate failure (measured=84.4660%, P2.1's own
    S8=85.1000%, diff=0.6340pp) -- ROOT CAUSE: wrong bundled checkpoint
    (augreg-v1) loaded regardless of the requested tag; corrected via a
    weight-loader override fix. RESOLVED and confirmed on real Colab
    hardware.** This is a SEPARATE question from item #35 above:
    FP-parity's `_fqvit_fp_pass` always runs with `quant=False` (no
    calibration, no fake-quant), so `BIT_TYPE_S`/`--lis` are irrelevant
    here -- the two investigations share a symptom (a ViT-B gap) but not
    a mechanism.

    **Root cause.** FQ-ViT's `vit_base_patch16_224()` factory loads its
    pretrained weights via a hardcoded call to its own
    `load_weights_from_npz(model, url, ...)`, always pointed at ITS OWN
    bundled augreg-v1 URL -- FQ-ViT's code has no tag/version concept, so
    this call resolves to the same weights every time regardless of which
    checkpoint TANDEM intends to compare against. `common/checkpoint_pin.py`'s
    override mechanism exists specifically to redirect this one call to
    TANDEM's pinned `augreg2_in21k_ft_in1k` checkpoint instead. The
    override itself had two real, sequential bugs (detailed below) that
    meant it silently never took effect on any run before this fix --
    every "Step B" measurement was in fact still on the stock augreg-v1
    checkpoint, indistinguishable from Step A's own run.

    **Same class of bug as the P2.1 ViT-B checkpoint-identity lesson,
    different mechanism.** `experiments/phase2_1_fleet/P2_tandem_pipeline.py`'s own
    `PUBLISHED_TOP1_LOOKUP` table and `RESOLVED_PRETRAINED_TAG` sanity
    gate exist because timm's OWN bare-name resolution for
    `vit_base_patch16_224` drifted across timm releases -- a real Colab
    run once silently resolved to `augreg2_in21k_ft_in1k` where an earlier
    run had resolved to a different tag, and P2.1 had to build explicit
    defenses (print the resolved tag; cover multiple plausible tags in the
    reference table) against exactly this failure mode: believing you are
    evaluating checkpoint X when the actually-loaded weights are
    checkpoint Y. This session's bug is the SAME CLASS -- a silent
    checkpoint-identity mismatch -- via a DIFFERENT MECHANISM: not timm's
    version-dependent bare-name resolution, but a Python monkey-patch that
    was installed without actually intercepting the real call site.

    **How the override failed to take effect (two separate bugs, both
    found and fixed in sequence):**
    1. `_FQVIT_OVERRIDE_SOURCE` originally patched only
       `models.utils.load_weights_from_npz` -- the function's defining
       module. But `models/vit_quant.py` (where the factory functions
       actually live) does `from .utils import load_weights_from_npz` at
       its own module level, which binds an INDEPENDENT local name,
       decoupled from `models.utils`'s own namespace from that point on --
       the factory functions call THEIR OWN local binding, never
       `models.utils`'s. Confirmed via a real run: the "Step B" measured
       value (84.4660%) was bit-for-bit identical to a deliberately
       unoverridden stock run, over the full 50,000-image set -- two
       different checkpoints producing identical accuracy to four decimal
       places is not possible by chance. Fixed by also patching
       `models.vit_quant.load_weights_from_npz` directly.
    2. That fix alone was still insufficient. `assert_override_fired`
       (the new safety net, below) caught a second failure on the very
       next real run -- same symptom (84.4660% again). `import
       models.utils` inside `sitecustomize.py` (which runs at Python
       interpreter startup) raised `ModuleNotFoundError`, because the
       real working directory wasn't yet reflected on Python's own
       `sys.path` at that point -- silently swallowed by a bare `except
       ImportError: pass`, so the whole override block quietly no-opped.
       Fixed by explicitly inserting `os.getcwd()` onto `sys.path` before
       the `models` import (the subprocess's real OS-level cwd, set by
       `subprocess.Popen(cwd=...)`, is always correct regardless of
       Python's own bootstrap timing). Verified this time by actually
       reproducing the bug -- a mock `models` package plus a fake `torch`
       stub, run under both invocation styles this project uses (plain
       script and `-m` module) -- confirming the old code fails and the
       new code succeeds in both, not by reasoning alone (the first fix
       had also been reasoning alone, and had been wrong).

    **Permanent safety net added, not just a one-off fix**:
    `common/checkpoint_pin.py`'s new `assert_override_fired(stdout,
    context_desc)` asserts the override's own confirmation print line
    actually appears in a subprocess's captured output, whenever the
    override was requested. Wired into every override-active call in BOTH
    `b3_fq_vit/run.py` and `b2_repq_vit/run.py` (added preemptively to B2
    too, even though B2's override was never actually broken -- the risk
    class is real for any future model/baseline using this mechanism, and
    the check costs nothing). This is what caught bug #2 above
    immediately, rather than requiring another round of numerical-
    coincidence detective work -- a checkpoint override silently failing
    to take effect can no longer happen undetected for either baseline.

    **Confirmed resolved on real Colab hardware**: corrected FP-parity
    gate: `harness=85.1540%  P2.1_S8=85.1000%  diff=0.0540pp (tol=0.15pp)
    -> PASS` -- consistent with a genuine augreg2 measurement (vs.
    augreg-v1's 84.53%, the value every broken run kept producing). Real
    Step-B W8A8 row written: `top1_50k=83.526`,
    `checkpoint_tag=vit_base_patch16_224.augreg2_in21k_ft_in1k` (correct
    pinned tag). Internally consistent with augreg2's higher FP accuracy
    producing a correspondingly higher quantized result (83.526 vs Step
    A's 82.630). `gate_status=not_applicable` by design (no published
    reference exists for FQ-ViT on this exact checkpoint); this row's
    validity rests on the now-passing FP-parity gate above.

    **For completeness, not causation**: a transform side-by-side
    (ruling item #1) was also run for real on Colab hardware and
    found FQ-ViT's stock transform (resize=248, bicubic, mean/std=0.5) to
    be an EXACT match against timm's resolved config for augreg2 --
    ruling out preprocessing as any part of the cause. A checkpoint-key
    diagnostic (152/152 keys matching) similarly ruled out a state-dict
    naming mismatch as a contributing factor. The deficit was entirely
    attributable to the weight-loader bug above.

    **B3's data collection is now structurally complete**: all 4 models
    (DeiT-T/S/B, ViT-B) x both Step A and Step B = 8/8 rows present in
    `results.csv`. The only remaining open item for B3 is item #35's
    methodology fork (8/8/4 vs 8/8/8), which affects the four Step-A
    `reproduction` rows' `gate_status`, not any Step-B `comparison` row --
    `compare_pareto.py`/B5 only reads `row_kind=comparison` rows (see item
    #35's own note), so this fork does not block B5 from reading a
    complete, correct `results.csv` for B3.

37. **B4 (sequential composites) rewritten per a 7-item spec:
    best-over-both-orders is now the defining mechanism, the
    operating-point grid is computed (not guessed) against DeiT-S's real
    BOPs formula, and the cost-ruler's two required composite reductions
    are proven and printed before any GPU work.** This supersedes item
    #10-11's original "2 hardcoded composites, never executed" state --
    the underlying integration-risk analysis in those items (the
    `Attention.forward` conflict and its resolution) is UNCHANGED and
    still governs; only the composite construction/grid/reporting layer
    on top of it was rewritten.

    **Best-over-both-orders (spec item #2, the defining requirement).**
    A staged pipeline has two orders (merge-then-quantize,
    quantize-then-merge) and the baseline must be the stronger of the two
    at each point, or a reviewer can dismiss the joint-vs-staged
    comparison as built on a weak baseline (the original framing,
    citing Kim et al.'s "sequential must be the max over orderings"
    lesson). `run.py` now runs BOTH orders for every operating point,
    writes BOTH as `row_kind=reproduction` rows (config includes the
    order suffix), and writes ONE `row_kind=comparison` row per point =
    max(both orders) -- the only row `compare_pareto.py`/B5 is meant to
    read. Both per-order measurements are always in `results.csv`, so the
    max is auditable, never silently kept as one order.
    `build_composite(order, ...)` implements both orders by swapping
    which of two composable steps (ToMe-patch-and-set-r,
    quantize-and-reparameterize) runs first -- verified this resolution
    strategy (skip RepQ's `build_model()`, use `quant_model()`'s generic
    Linear-wrapping) is genuinely order-independent: `tome.patch.timm`'s
    class-level reassignment and `quant_model()`'s Linear-level wrapping
    operate at different levels of the module hierarchy and do not
    interfere regardless of which runs first.

    **Degenerate-order detection, not a shortcut.** When an operating
    point's quantization stage is a no-op (w_bits==a_bits==16) or its
    merge stage is a no-op (r==0), the two orders are mathematically
    IDENTICAL (composing with a no-op stage in either position gives the
    same model) -- `orders_coincide()` detects this and runs the
    underlying model ONCE (a second GPU pass would produce a bit-identical
    number), still writing BOTH per-order rows with the degeneracy noted
    explicitly, plus the comparison row. Same principle as this project's
    existing "Step A == Step B, don't run twice" checkpoint-coincidence
    pattern, applied to order-identity instead.

    **Operating-point grid (spec item #4), computed not guessed.** Grid
    search over ToMe's r (already-exercised range, extended within the
    same general closed-form schedule -- not a new mechanism) x RepQ's
    already-validated bit-widths {16="off", 8, 6} (deliberately NOT
    widened to untested bit-widths, e.g. asymmetric w8a16 -- a real
    tradeoff, resolved by explicit user decision this session), against
    DeiT-S's real BOPs formula and its real `budget_fractions` (read live
    from the canonical P2.1 summary, asserted equal to a hardcoded copy,
    never silently trusted alone). Result: 5 of 6 points land within
    ~1.8% of their target, the 6th (41%) within 1.14% after widening the
    r-search range specifically for it. A real, honestly-reported
    consequence of only using validated bit-widths: 3 of 6 points (the
    higher-budget ones) end up as pure-ToMe (w16a16, no quantization at
    all) rather than genuine composites, since 8-bit quantization alone
    already drops BOPs to ~31% of C_id, leaving nothing in the validated
    bit-width set to fill the 31-100% range except unquantized merge
    ratios. Documented in `run.py`'s own module docstring with the full
    search table, not hidden.

    **Cost-ruler reductions (spec item #3).** Two new shared functions in
    `common/bops_identity_check.py` (not B4-local -- the same proof
    applies to any future composite baseline):
    `check_composite_r0_reduces_to_ptq` (composite-at-r=0 BOPs ==
    `uniform_bits_trajectory`'s pure-PTQ BOPs, for the same bits) and
    `check_composite_w16a16_reduces_to_tome` (composite-at-w16a16 BOPs ==
    `tome_schedule_n_l`'s pure-ToMe BOPs, for the same r). Both run for
    real (not just imported) as part of this session's verification:
    ```
    composite-at-r=0 (w8a8) BOPs=3.592760e+11 == pure-PTQ (w8a8) BOPs -- PASSED
    composite-at-w16a16 (r=13) BOPs=7.241061e+11 == pure-ToMe (r=13) BOPs -- PASSED
    ```
    `run.py::main()` prints these (plus the identity check) BEFORE any GPU
    work, every real invocation -- spec item #7's "report before full
    runs" requirement, satisfied structurally, not just as a one-time
    demo.

    **Single source of truth (spec item #1).** RepQ-ViT is the composite
    PTQ partner (stated explicitly in `run.py`'s docstring, per the
    documented reasoning: the stronger/cleaner PTQ method of B2/B3); a
    future FQ-ViT composite family would be separately labeled, never
    blended in. `assert_no_forked_algorithm_logic()` -- a real, running
    static AST check, not a docstring claim -- asserts this file defines
    no function shadowing ToMe's or RepQ-ViT's own core algorithm entry
    points (`bipartite_soft_matching`, `merge_wavg`, `apply_patch`,
    `quant_model`, `set_quant_state`, `build_model`); verified to actually
    run and pass (see verification note below).

    **Completeness (spec item #6): `PLAN` is the declared per-model plan**
    (`n_operating_points=6`, `rows_per_operating_point=3` -- 2
    reproduction + 1 comparison per point -- `expected_results_csv_rows=18`)
    the notebook's completeness cell now derives `EXPECTED_ROWS` from
    (read via a subprocess through the pinned venv, since `PLAN` lives in
    a torch/timm-importing module), replacing the prior hardcoded `4`.

    **ViT-B: explicitly deferred, not attempted (spec item #5's ViT-B
    mention).** The original spec mentions ViT-B's Step-B checkpoint
    override as part of B4's gates; this pass is DeiT-S only, per the
    "paste DeiT-S first" instruction and an explicit
    scope decision made with the user this session (real additional work:
    ViT-B's own operating-point grid computed the same way, its own
    FP-parity/checkpoint-override wiring) -- not silently dropped, an
    acknowledged, deliberate follow-up.

    **Verification this session (no GPU available here, so everything
    checkable without one was actually checked, not just read):** the new
    `common/bops_identity_check.py` functions were RUN (not just
    imported) and both passed against DeiT-S's real numbers. `run.py`
    itself was verified by actually IMPORTING it (via minimal fake
    `torch`/`timm`/`tqdm` stub packages in a scratch directory, the same
    technique that caught the two real checkpoint-override bugs earlier
    this session) and running its torch-free logic for real:
    `assert_no_forked_algorithm_logic()` passed; `orders_coincide()` +
    `config_key_for()` produced the expected keys and degeneracy flags for
    all 6 operating points; `PLAN["expected_results_csv_rows"]` computed
    to 18 as expected; `print_cost_ruler_reductions()` and
    `print_operating_point_grid()` both ran and printed correct output
    against the real canonical C_id. `common/verify_colab_thin_wrappers.py`
    confirmed the updated notebook stays a thin wrapper; the regenerated
    `.ipynb` validated clean via `nbformat.validate`. What could NOT be
    verified without a GPU: the actual composite construction/calibration/
    evaluation path (`build_composite`, `_quantize_and_reparam`,
    `evaluate_one_order`) -- these need a real Colab or university-GPU run,
    same limitation as every other GPU-dependent change this project has
    made. Treat DeiT-S's first real run as a validation pass per item
    #10-11's original framing, which still applies to this new code just
    as it did to the old.

38. **B4's attention-matmul quantization scope is a real methodological
    difference from plain RepQ-ViT, and it makes the composite slightly
    OPTIMISTIC, not neutral -- stated explicitly (
    follow-up, item #3).** Per `run.py`'s own module docstring (the
    `Attention.forward` conflict section), RepQ-ViT's `build_model()` --
    which installs `MatMul` submodules so `quant_model()` can fake-quantize
    the Q@K/Attn@V matmuls -- is skipped entirely to keep ToMe's merge-
    metric contract intact. Consequence: B4's composite quantizes only
    qkv/proj/fc1/fc2 (the Linear layers), and leaves attention itself in
    full precision, throughout every operating point. Plain B2 (RepQ-ViT,
    via `build_model()`) DOES quantize attention. This means B4's measured
    accuracy at a given LABELED bit-width (e.g. "w8a8") reflects LESS
    actual quantization than the label alone implies, relative to what
    plain RepQ-ViT measures at the same label -- the composite is
    therefore systematically a little easier on itself than a
    fully-quantized w8a8 model would be, i.e. its accuracy numbers should
    be read as a slightly OPTIMISTIC upper bound for "true" w8a8 staged
    performance, not a neutral one. (This does not distort the composite's
    own BOPs accounting, which already only charges the Linear layers'
    bit-width and prices attention matmuls at 16x16 uniformly, matching
    TANDEM's own cost model exactly -- see item #37's cost-ruler section --
    so the x-axis is unaffected; only the y-axis (accuracy) carries this
    optimism.)

39. **41%/57.2%/80% operating-point audit (follow-up,
    items #1-2): the "strongest config at this budget" search was run
    for real, and it contradicts the original working hypothesis --
    reported precisely, not silently reconciled.** The prior working hypothesis
    was that a w8a8-with-small-r composite would dominate the
    pathological r=22/w16a16 point at 41% (final-block token count
    collapses to 4, an 8.1pp accuracy cliff vs the 57.2% point). An
    exhaustive grid search (r=0..40, bits in {16,8,6}, the SAME validated
    set established earlier this session -- deliberately not widened to
    untested asymmetric bit-widths) found this premise does not hold: w8a8
    tops out at 30.91% of C_id and w6a6 at 20.83%, BOTH at r=0 (the
    least-reduction, maximum-BOPs case for that bit-width) -- since any
    r>0 only lowers BOPs further, NEITHER bit-width can reach anywhere
    near 39%, 55.2%, or 78% of C_id. This is the same structural gap
    already documented in item #37's operating-point-grid section (the
    reason 3 of 6 points are pure-ToMe in the first place), now confirmed
    to also rule out a "smaller-r quantized" fix at these SPECIFIC three
    budgets specifically, not just observed as a general pattern.
    - **41% +/-2% (39.0-43.0%)**: exactly two candidates exist, both
      w16a16: r=22 (current, BOPs=41.47%, final-block token count 4) and
      r=23 (BOPs=40.06%, final-block token count ALSO 4) -- r=23 merges
      slightly MORE aggressively than r=22 and shows the identical
      late-block collapse signature; it is not expected to resolve the
      pathology, only to confirm (or, less likely, refute) that it is
      structural to this budget rather than specific to r=22. NOT YET RUN
      -- pending a decision (see below) on whether running it is worth
      the GPU time given this prediction.
    - **57.2% +/-2% and 80% +/-2%**: ZERO alternative candidates exist in
      either window at any bit-width -- the current r=15 and r=7 points
      are the ONLY grid members within tolerance. Item #2's audit
      therefore has an unambiguous, fully-tested answer for both: no
      smaller-r quantized alternative exists to run. Recorded here so the
      record shows the check was performed, per the standing
      instruction ("if no, record the check result").
    - **Open decision, raised rather than resolved unilaterally**: a
      genuine fix for 41%'s pathology (something that actually reaches
      that budget with real quantization headroom) requires either
      accepting r=22 as a real, reported structural limit of the
      validated-bit-widths-only staged baseline at this budget (not
      "superseded" by anything, since nothing better exists within the
      current constraints -- itself a citable finding about where a naive
      uniform staged approach breaks down), or revisiting the asymmetric
      bit-width option (e.g. w8a16/w16a8) explicitly declined earlier
      this session in favor of "accept as-is." Not decided here.

40. **B4 closure, DeiT-S (follow-up, supersedes item #39's
    open decision -- ruled that the structural cause is the answer,
    not something to fix by relaxing constraints). New analysis script
    `b4_sequential_composites/analyze_deit_small.py` (torch-free,
    pandas/numpy only, run via `python
    experiments/phase2_2_baselines/b4_sequential_composites/analyze_deit_small.py`) implements
    items #2-5 and is the single source of truth for every number below --
    all of them were RUN, not typed, against the real local `results.csv`
    and P2.1's real `plans.csv`.**

    **Item #1: r=23 audit point.** `run.py` refactored (`process_point()`
    extracted from the former inline per-operating-point loop, now shared
    by both the 6-point grid and a new `AUDIT_POINTS` list) and a new
    `--only-audit-r 23` CLI path added: order-degenerate (w16a16), one GPU
    pass, writes both `row_kind=reproduction` rows (per this project's
    standing degenerate-order policy) and deliberately NO
    `row_kind=comparison` row, so `compare_pareto.py`/B5 never reads it --
    evidence for whether r=22's pathology is r-specific, not a competing
    entry in the canonical 6-point set. Verified via the same faithful-stub
    technique used earlier this session (fake `torch`/`timm`/`tome`/`utils`/
    `quant` packages matching the real upstream export structure): the
    audit point's control flow (order-degenerate detection, `apply_patch`
    call, exactly 2 rows written, no comparison row) confirmed correct;
    the actual composite construction/eval still needs a real GPU pass
    (not yet run -- pending, see the r=23 line requested for paste-back).

    **Item #2: structural cost split -- verified bit-for-bit, not just
    algebra on paper.** DeiT-S's BOPs formula (`common/bops.py`) prices
    the linear term proportional to `bw*ba` and the attention term as a
    FIXED constant regardless of bit-width -- solving `L8 + A =
    frac(W8A8@r0)` and `4*L8 + A = 1.0` (since `L(16,16) = 4*L(8,8)`, as
    256/64=4) gives the split directly. Computed AND independently
    cross-checked (the derived split predicts W6A6@r0 and W8A16@r0 exactly,
    to 1e-9, without using either to derive it):
    ```
    linear fraction (W16A16@r0)   = 92.1232%
    attention fraction (constant) =  7.8768%
    W8A8@r0  = 30.9076%
    W6A6@r0  = 20.8317%
    W8A16@r0 = 53.9384%  (asymmetric, illustrative only -- see item #39, still excluded)
    ```
    Matches the pre-registered figures (92.1%/7.9%/30.91%/20.83%/53.9%)
    exactly at the stated precision. Verbatim record of the finding this
    verifies: *"DeiT-S cost splits 92.1% linear / 7.9% attention (derived
    from W8A8@r0=30.91% and W6A6@r0=20.83%, cross-checked). Uniform-bit
    staged quantization therefore spans only 0-31% of C_id; the 31-54%
    range is reachable only via asymmetric precision (W8A16@r0=53.9%),
    which no published uniform PTQ baseline validates and is excluded on
    that principle. Above 31%, uniform-r merging is the sole lever; the
    41% budget forces r=22-23 with final blocks at 4 tokens (top1 70.21).
    Exhaustive search r=0..40 x bits{16,8,6} found no alternative within
    +/-2%."* This closes item #39's open decision: the ruling is
    (a), not (b) -- r=22 stands as a real, reported structural limit, not
    superseded by anything; asymmetric bit-widths remain excluded.

    **Item #3: attention-skip bias, corrected.** Observed floor (B4's
    `r2_w8a8_best` comparison row, 79.704%, vs B2's plain `w8a8`
    comparison row, 79.652%, same model, same nominal bit-width, only
    difference is B4's attention-quantization skip -- item #38): **+0.052pp
    exactly**, matching the pre-registered figure to 3 decimal places.
    Estimated total range (0.08-0.15pp, including foregone r=2 merge
    damage this floor alone can't isolate) is a documented estimate,
    recorded as given, not independently re-derived. Field:
    `composite_attention_skip_bias_pp = {floor: 0.052, est_range: [0.08,
    0.15]}` (written into `B4_deit_small_summary.json` by the analysis
    script). Confirms staged's apparent edge over TANDEM at the 3 low
    budgets (item #4/#5 below) is partially, not wholly, attributable to
    this -- the 15% delta (-0.008pp) is smaller than the floor alone, but
    21%'s (-0.566pp) and 29.3%'s (-0.471pp) are not.

    **Item #4: staged Pareto-frontier dominance -- the headline finding,
    proven by assertion, not just plotted.** Built the staged Pareto set
    from B1's r=0 anchor (`tome/deit_small_patch16_224/r0`, off-the-shelf,
    79.826% @ exactly C_id) plus the 6 real B4 comparison rows. Two
    assertions, both PASSED against the real data:
    1. `{r22_w16a16_best, r15_w16a16_best, r7_w16a16_best}` (the 41/57.2/80%
       points) are each dominated by some cheaper-and-at-least-as-accurate
       point on the frontier.
    2. The STRONGER, more specific claim: `r2_w8a8_best` (29.3%) ALONE
       dominates all three simultaneously (single reference point, not
       "some point or other" per point) -- BOPs and top1 both checked
       directly: 3.345e11 <= {4.820e11, 6.597e11, 9.222e11} and 79.704% >=
       {70.206%, 78.300%, 79.564%} in every case.
    Staged Pareto frontier: `{r17_w8a8_best, r10_w8a8_best, r2_w8a8_best,
    r0}` -- i.e. **the staged frontier saturates at 29.3% of C_id**;
    spending more budget than that, within the validated-bit-widths-only
    design, makes the staged baseline's OWN best achievable accuracy
    WORSE, never better. Field `staged_frontier_saturation` records the
    saturating config and the dominated set.

    **Item #5: programmatic comparison table + figure.** TANDEM's real
    DeiT-S Pareto points are read from `plans.csv` via
    `compare_pareto.load_tandem_pareto` (unchanged, existing function --
    no new parsing logic), never typed. One real bug caught and fixed
    while building this: an early version of the script assumed P2.1's
    own highest-C plan point IS exactly C_id (the "identity plan"
    reasoning) -- a real assertion firing on real data proved this false:
    P2.1's own 80%-budget plan only achieved 76.97% of C_id in practice
    (`budget_utilization=0.9621` in the canonical summary), so its curve's
    real max sits well short of full C_id. Fixed to read `c_id` directly
    from the canonical summary instead of inferring it from the curve's
    own max -- this is also independent, direct confirmation of why the
    80% row is flagged `extrapolated=True` below (B4's r=7 point, at
    79.34% of C_id, genuinely exceeds TANDEM's own highest real
    measurement on this curve).
    ```
    config               staged_bops  staged_top1  tandem_top1  delta(T-S)  bias_flag  extrap
    r17_w8a8_best         1.7452e+11       77.434       77.426      -0.008       True   False
    r10_w8a8_best         2.4336e+11       79.272       78.706      -0.566       True   False
    r2_w8a8_best          3.3450e+11       79.704       79.233      -0.471       True   False
    r22_w16a16_best       4.8203e+11       70.206       79.601      +9.395      False   False
    r15_w16a16_best       6.5966e+11       78.300       79.663      +1.363      False   False
    r7_w16a16_best        9.2223e+11       79.564       79.826      +0.262      False    True
    ```
    **[Superseded by item #42's item 3]** The `r17_w8a8_best` row's
    -0.008pp delta is smaller in magnitude than the +0.052pp
    attention-skip-bias floor measured in item #3 above -- reclassified
    there as a tie within measurement bias, not a real staged advantage
    over TANDEM at the 15% budget. See item #42.

    Figure: `B4_deit_small_pareto.png` -- TANDEM's curve (line), staged
    frontier points (filled squares), staged dominated points (grey X,
    shape-coded not just color-coded so the distinction survives
    grayscale/CVD), matching this project's existing P2.1 figure style
    (`experiments/phase2_1_fleet/P2_tandem_pipeline.py`'s cross-model overlay: figsize=(9,6),
    "o-" markers, dpi=150) for visual consistency.

    **Item #6: scope ruling, logged.** B4 closes at DeiT-S -- the
    G1/G1-HW model, where the staged comparison is load-bearing.
    Composites on Tiny/Base/ViT-B are DEFERRED post-G1-HW as optional
    strengthening. This is a deliberate scoping decision, recorded here
    for the record, not an omission discovered later.

41. **B5 (compare_pareto) final P2.2 closure pass (
    follow-up, subsumes item #40's outstanding B4-closure items). New
    script `b5_final_closure.py` (torch-free, pandas/numpy/matplotlib) is
    the single source of truth for every number and assertion below, run
    for real against the local `results.csv` and all four models' real
    `plans.csv` -- nothing typed.**

    **Part A (finish the B4 record).** Items A.2's fields (structural
    cost-split verbatim block, `composite_attention_skip_bias_pp`,
    `staged_frontier_saturation`) were already written into
    `deviations.md`/`B4_deit_small_summary.json` in item #40 above --
    confirmed still present and unchanged. **Item A.1, the r=23 w16a16
    audit GPU pass, is now RUN (real Colab/Tesla-T4 data)**: order-
    degenerate (w16a16, one GPU pass, both order rows record the same
    measurement per the standing coincide-don't-double-run policy),
    `top1_50k=68.096%`, `n=50000`, `bops_ours=4.656647e+11` (32.13% of
    C_id), `n_l_per_block=[197, 174, 151, 128, 105, 82, 59, 36, 19, 10,
    6, 4]`. This confirms the r=22 pathology (final-block token count 4,
    top1 collapsing to 70.206%) is **structural to the 41% budget, not
    r=22-specific** -- r=23's final-block count is also 4 and its top1
    (68.096%) is worse still, consistent with the exhaustive r=0..40 grid
    search in item #39 finding no escape from this regime at that budget.
    Two `row_kind=reproduction` evidence rows written
    (`r23_w16a16_merge_first`/`r23_w16a16_quant_first`), deliberately no
    `row_kind=comparison` row (per item #40's item #1 design), so this
    point does not enter B5's comparison-only view or the Pareto/table
    outputs below -- confirmed by re-running `b5_final_closure.py` after
    this row landed: every assertion, count, and figure is unchanged.

    **Part B (consumption rules, asserted not assumed).**
    `assert_row_filter()` reads `compare_pareto.comparison_eligible_rows()`
    (unchanged existing filter: `row_kind == "comparison"` AND
    `setting` in `{"", "off_the_shelf"}`) and asserts, against the real
    data: counts `{deit_tiny: 3, deit_small: 12, deit_base: 3,
    vit_base: 3}` exactly; and separately, by direct query (not just "by
    construction"), that zero `setting=trained_with_tome` rows and zero
    per-order (`merge_first`/`quant_first`) composite rows appear under
    `row_kind=comparison` -- both are structurally always
    `row_kind=reproduction`, verified rather than assumed. Both
    assertions PASSED. TANDEM's DeiT-S curve is read programmatically
    from `plans.csv` via the existing `compare_pareto.load_tandem_pareto`
    (same function item #40 used, no new parsing logic) and printed
    verbatim for independent cross-check.

    **Part C outputs.**
    - *Pooled-baseline Pareto front* (`pooled_pareto_front()`): same
      dominance definition as item #40's item #4, pooled across ALL 4
      methods' `row_kind=comparison` rows per model (not just B4's).
      DeiT-S's computed front -- `{r0, r10_w8a8_best, r17_w8a8_best,
      r2_w8a8_best, w6a6}` -- was asserted equal (as a set) to the
      independently pre-registered front and PASSED. Saturation
      ceiling confirmed: 79.704% at 28.78% of C_id, strictly below 100%.
    - *4-panel figure* `B5_all_models_pareto.png`: one panel per model,
      x=BOPs (log scale), y=top1_50k; points colored by method with
      black edges on frontier points, dominated points drawn as grey X's
      (shape-coded, matching item #40's grayscale/CVD-safety choice);
      pooled front drawn as a red dashed step line; TANDEM's own curve
      overlaid in black; DeiT-S panel annotates the 79.704%/28.78%
      saturation point.
    - *Comparison table* (21 rows, all 4 models): each baseline
      comparison row plus its nearest-budget TANDEM point, both sides'
      REAL BOPs, delta, and three flags -- `attention_skip_bias_flag`
      (composite w8a8 rows, per item #38), `cost_under_credited_flag`
      (all `fq_vit` rows -- FQ-ViT quantizes LayerNorm/Softmax/attention
      beyond the four-Linears cost ruler, so its reported BOPs
      under-represents its real quantization work; documented
      parenthetical recorded verbatim as the flag's description, not
      independently re-derived), `dominated_flag` (from the pooled
      front above).
    - `compare_summary.json`: per-model pooled fronts, the DeiT-S
      saturation fields, the full flagged comparison table, row-filter
      assertion results, a manifest hash (`41ddeb0ebb2fea82` as of this
      run) of the whole summary, and `plans_csv_provenance` -- each
      model's `plans.csv` path plus a `sha256_16` content hash, so a
      reviewer can verify TANDEM's numbers weren't hand-edited after the
      fact. **One issue caught and fixed during self-review before
      shipping**: the first version stored each `plans.csv` path as an
      absolute local filesystem path (`E:\EDGE AI\tandem\notebooks\...`),
      which is meaningless off this machine and not appropriate for a
      reviewer-facing artifact; fixed to store the path relative to the
      repo root (`experiments/phase2_1_fleet/results/P2/<model>/plans.csv`) instead,
      re-run to confirm the hash and every downstream assertion are
      unaffected.
    - `experiments/phase2_2_baselines/reviewer_release/` (see REMOVAL_CANDIDATES.md -- a duplicate, kept but flagged) folder created, containing
      `B4_deit_small_pareto.png`, `B5_all_models_pareto.png`,
      `compare_summary.json`, and `results.csv` -- the self-contained
      handoff package for reviewer review.

    **Two more code-quality issues caught and fixed during self-review**
    (neither affected correctness of any number already reported, both
    caught before shipping): a debug print inside the per-model loop
    used the bare `config` string instead of the full `method/config`
    label when printing frontier/dominated sets -- ambiguous, since
    `repq_vit` and `fq_vit` both use `"w8a8"` as their `config` value;
    fixed to use the full label. And a leftover unused variable
    (`deit_small_front_configs`) plus an unused module-level constant
    (`EXPECTED_DEIT_SMALL_FRONT`, not actually wired into the assertion
    it was meant for -- a redundant local list was hardcoded instead)
    from an earlier draft; fixed by removing the dead variable and
    wiring the constant into the assertion it names.

42. **P2.2 record patch (follow-up, items 2-3; item 1 is
    the cap-formula correction folded into item #32 above). Documentation
    + narrative-flag additions only -- no GPU runs, computed from the
    already-real `results.csv`/comparison table by re-running
    `b5_final_closure.py`.**

    **Item 2: `quantizer_orthogonality`.** RepQ-ViT beats TANDEM at BOTH
    of its matched-BOPs cost points (w6a6, w8a8), at ALL 4 models, in the
    real comparison table (`delta_tandem_minus_baseline_pp < 0` in all 8
    RepQ-ViT rows -- computed and asserted in `b5_final_closure.py`, not
    counted by hand): range **[-2.65, -0.1588] pp**, largest at
    `deit_base_patch16_224`/w6a6 (-2.65pp) -- matches the pre-registered
    range and location exactly. Recorded interpretation, per the
    documented ruling: this reflects a **quantizer-quality difference**
    (RepQ-ViT's reparameterized PTQ pipeline vs TANDEM's own simple
    per-block static symmetric quantizer -- `experiments/phase2_1_fleet/P2_tandem_pipeline.py`'s
    `_quantize_weight_rows`/`_quantize_tensor`), **not allocation regret**:
    uniform-bit, r=0 configurations already lie inside TANDEM's own
    `(r, per-block-bits)` plan search space (TANDEM's allocator explores r
    starting from 0, so it is not structurally barred from this region --
    RepQ-ViT simply extracts more accuracy from a given nominal bit-width
    than TANDEM's current quantizer does). Recorded as a stated limitation
    of TANDEM's current quantizer implementation, not of its
    cost-allocation search, with **composability** (pairing TANDEM's
    allocation search with a stronger per-block quantizer such as RepQ-
    ViT's reparameterization) named as orthogonal future work. Field
    `quantizer_orthogonality` (note, `delta_range_pp`, `max_delta_point`,
    `stated_limitation`, `future_work`) written into `compare_summary.json`,
    computed programmatically from the real `repq_vit` comparison rows
    (asserted `== 8` rows: 4 models x 2 cost points) -- nothing typed.

    **Item 3: 15%-budget delta reclassified as a tie within measurement
    bias.** The composite `r17_w8a8_best` row's delta vs TANDEM is
    `-0.0084pp` (item #40/#41's tables) -- smaller in magnitude than B4's
    own measured attention-skip-bias FLOOR (`+0.052pp`, item #40's item
    #3, read programmatically from `B4_deit_small_summary.json`'s
    `composite_attention_skip_bias_pp.floor`, never re-typed). Since the
    entire observed delta at this one point is smaller than the known,
    already-quantified measurement bias favoring staged composites at
    w8a8, this specific point is **not evidence of a real staged
    advantage over TANDEM at 15%** -- it is a tie once that bias is
    accounted for. New computed (not hardcoded) column
    `tie_within_measurement_bias_flag` added to the comparison table:
    `True` only where `attention_skip_bias_flag` is set AND
    `abs(delta) <= attention_skip_bias_floor_pp`. Verified against the
    real table: exactly one row qualifies --
    `(deit_small_patch16_224, r17_w8a8_best, -0.0084)` -- none of the
    other w8a8 composite rows (`r10_w8a8_best` at -0.5663pp,
    `r2_w8a8_best` at -0.4706pp) come close to the 0.052pp floor, so they
    remain real (if bias-flagged) TANDEM-favorable deltas, not ties. The
    `attention_skip_bias_floor_pp` value itself is also now written as a
    top-level field into `compare_summary.json` for provenance.

    Re-ran `b5_final_closure.py` after both additions: all pre-existing
    assertions (row-filter counts, DeiT-S pooled front, saturation
    ceiling) still PASS unchanged; `reviewer_release/` re-synced.
