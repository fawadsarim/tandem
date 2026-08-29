# requirements/ — reference copies, not the live install source

These 8 files are **read-only reference copies** for at-a-glance comparison
across every environment this project uses. **No install command reads
from this directory.** Each environment's actual `pip install -r
requirements.txt` reads the live file sitting next to that environment's
code — listed below. If you edit a pin, edit the live file; this directory
is not regenerated automatically and can drift if forgotten.

| Reference copy | Live source (what `pip install -r` actually reads) | Environment |
|---|---|---|
| `root.txt` | `/requirements.txt` | Repo-root convenience venv (plotting/analysis only, no training/eval) |
| `baselines_shared.txt` | `experiments/phase2_2_baselines/requirements.txt` | Shared across all P2.2 baseline orchestration |
| `b1_tome.txt` | `experiments/phase2_2_baselines/b1_tome/requirements.txt` | B1 (ToMe) isolated venv |
| `b2_repq_vit.txt` | `experiments/phase2_2_baselines/b2_repq_vit/requirements.txt` | B2 (RepQ-ViT) isolated venv |
| `b3_fq_vit.txt` | `experiments/phase2_2_baselines/b3_fq_vit/requirements.txt` | B3 (FQ-ViT) isolated venv |
| `b4_sequential_composites.txt` | `experiments/phase2_2_baselines/b4_sequential_composites/requirements.txt` | B4 (sequential-composite) isolated venv |
| `jetson_ondevice.txt` | `jetson/requirements.txt` | Runs ON the Jetson itself (torch deliberately absent — see that file's own header, JetPack ships its own wheel) |
| `jetson_host.txt` | `jetson/requirements-host.txt` | Runs on a HOST/Colab GPU for jetson/'s export + HOST-side eval scripts |

## These environments are NOT compatible with each other

Torch pins alone span four incompatible major/minor versions across this
project, by design, not oversight:

| Environment | torch pin | Why |
|---|---|---|
| `root.txt` | unpinned (`torch`) | Convenience-only venv, no numeric-sensitive work |
| `b1_tome` | `torch==1.12.1` | ToMe's own documented floor (needs `scatter_reduce`) |
| `b2_repq_vit` | see `b2_repq_vit.txt` | Pinned to RepQ-ViT's own tested environment |
| **`b3_fq_vit`** | **`torch==1.7.1`, `torchvision==0.8.2`** | **FQ-ViT needs a genuinely old torch — this baseline is intentionally isolated in its own venv, never installed alongside any other baseline's environment.** |
| `b4_sequential_composites` | see `b4_sequential_composites.txt` | Composes B1+B2's pinned upstream commits, own venv |
| `jetson_host.txt` | `torch==2.11.0` | Export/host-side eval scripts, unrelated to any baseline's pin |
| `jetson_ondevice.txt` | absent by design | JetPack ships its own CUDA-matched torch wheel; this file explicitly does not attempt to pin one — see the file's own header comment |

**Never install two of these requirement sets into the same environment.**
Each baseline (`b1_tome`/`b2_repq_vit`/`b3_fq_vit`/`b4_sequential_composites`)
and each jetson context (on-device/host) is designed to run in its own
isolated venv — see `docs/REPRODUCING.md` Tier 2 for the exact per-stage
environment setup commands.
