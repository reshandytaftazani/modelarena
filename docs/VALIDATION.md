# Validation evidence

## Day 1 — 2026-10-22

Scope authorized by the user's request to implement Day 1 only. Day 2 onward, frontend scaffolding, training, benchmarks and dataset downloads were not executed. The user confirmed 2–3 focused hours per day.

Environment observed: Windows 11 Home Single Language build 22621, AMD Ryzen 5 5500U, 6 physical / 12 logical processors, 16,455,942,144 bytes (15.326 GiB) usable physical memory. The inventory sample at 12:16 UTC recorded 4,186,640,384 bytes (3.899 GiB) **available** memory. This is a changing snapshot, below the future 4 GiB benchmark reserve, not evidence that a 4 GiB child is safe to start. Power scheme was Balanced, AC connected; Windows Settings power mode was not observed separately.

Toolchain: Python 3.12.15 (uv managed), uv 0.13.5, existing Node 24.19.0 and npm 12.2.0. The architecture's proposed Node 22 was provisional; Day 1 pins the installed Node 24 toolchain. React/Vite compatibility remains a Day 22 check. No existing Git repository was reinitialized. Work was performed in the requested checkout.

Locked direct versions: FastAPI 0.159.1, Pydantic 2.14.1, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, torch 2.14.1+cpu, MLflow 3.17.2, psutil 7.2.2, uvicorn 0.54.0, python-multipart 0.0.32, threadpoolctl 3.7.0, pytest 9.1.1 and httpx 0.28.1. All installed from wheels; torch uses the explicitly scoped CPU index. The full version/hash graph is in `uv.lock`; frontend packages will be locked on Day 22. Windows x86-64 is the verified target; other platforms are unverified.

Executed evidence:

- Initial `uv lock`, dependency installation into a new `.venv`, and final `uv sync --locked` succeeded. Direct pins were frozen from that resolution, then the lock was regenerated and locked sync repeated successfully.
- `uv run --locked python -c "import fastapi, sklearn, torch, mlflow; print(torch.__version__)"` and an expanded smoke check succeeded. torch reported `2.14.1+cpu`, `torch.version.cuda is None`, `torch.cuda.is_available() == False`; a CPU tensor calculation returned `[2.0, 2.0]`. No training ran.
- `npm.cmd ci --ignore-scripts` succeeded at repository root: zero package dependencies and zero audit findings. This verifies the toolchain manifest only. The planned `apps/web` install is deferred to Day 22.
- Settings tests were written before implementation; their first run failed because `apps.api.settings` did not exist. Hardware-inventory tests first failed because `scripts.profile_hardware` did not exist. After implementation, `uv run --locked pytest -q` passed **53 tests**, with no skips. A fresh reviewer then found an iCloud Drive denylist typo. The new sync-folder regression reproduced it (1 failed, 57 passed); the spelling was corrected and the expanded suite rerun. Tests exercise strict policy validation, immutable settings, local home resolution, environment inputs, available-memory reporting and lock identity.
- `uv run --locked python -m scripts.profile_hardware` succeeded, matched the confirmed hardware, and wrote the private `inventory/hardware.json`. Settings loads do not create directories. Inventory files are observations, not benchmark evidence.
- `uv run --locked mlflow gc --help` confirmed the pinned release exposes permanent deleted-run metadata/artifact cleanup with explicit `--backend-store-uri` and `--run-ids`. The CLI warns invalid artifact URLs can bypass artifact cleanup. Actual SQLite/artifact purge correctness remains a Day 12/25 acceptance test; help output alone does not prove deletion.
- `git check-ignore` confirmed representative secrets, uploads/data, models/artifacts, databases, runtime inventory, virtualenv and frontend dependencies are ignored. `git diff --check` passed.

Policy version 1 defaults remain provisional: T=4, RSS=4 GiB, one active job, serial CV, estimator n_jobs=1, torch inter-op=1 and DataLoader workers=0. Configuration rejects unreviewed cap increases; it does not yet enforce runtime resource limits. Thread policy, cancellation, spawn behavior and full 2/4/6 performance comparison remain future work. No experiment-success claim is made.

Execution decisions: Day 1 uses the installed Node 24 rather than the provisional Node 22 (frontend compatibility must be verified on Day 22). The root npm lock pins only the toolchain, preserving the plan's frontend deferral. The existing checkout was used for this setup work. The implementation and evidence were initially left uncommitted for user review; the dated entry below records subsequent commit verification and delivery. No P0 requirement was removed.

Final review fix: `test_known_sync_folders_cannot_host_active_storage[iCloudDrive]` failed before the denylist correction and passed afterward. Final `uv run --locked pytest -q`: **59 passed in 2.50 s, zero skips**. `uv lock --check` and `git diff --check` also passed. No review findings remain deferred.

## Day 1 commit verification — 2026-10-23

The user explicitly requested the four planned Day 1 commits. Existing implementation was packaged in dependency order, with no training, benchmark, frontend scaffold or Day 2 implementation. The prior Day 1 results above remain historical evidence.

| Commit | Delivered scope | Fresh verification |
|---|---|---|
| `2dc1dc7` | Locked Python/Node toolchain, CPU torch source, root npm manifest and local-workspace ignore rule | Exported staged tree: `uv lock --check`, `uv sync --locked`, CPU import smoke check and `npm.cmd ci --ignore-scripts` all exited 0. Fresh environment installed 105 packages; npm reported zero vulnerabilities. |
| `f989c16` | Immutable resource/run contracts, local storage settings and owning tests | Exported cumulative staged tree: `uv run --locked pytest tests/unit/test_settings.py -q` — 58 passed in 0.53 s, zero skips. |
| `de54447` | Hardware inventory command and owning test | Exported cumulative staged tree: inventory test — 1 passed in 3.24 s; full suite — 59 passed in 3.41 s, zero skips; inventory CLI exited 0. |

The staged trees were exported with `git checkout-index` to a new temporary directory outside the repository, using a separate freshly synchronized virtual environment. Only staged files and prior commits were exported; later unstaged source could not satisfy earlier checks. The export has no `.git`, so its inventory correctly reported code revision unavailable. No source or dependency versions were changed during packaging.

CPU smoke command executed on the first staged tree:

```powershell
uv run --locked python -c "import fastapi, sklearn, torch, mlflow; print(torch.__version__); assert torch.version.cuda is None; assert not torch.cuda.is_available()"
```

It reported `2.14.1+cpu` and exited 0. Root npm verification covers the toolchain manifest only, with no frontend dependencies. Representative `.env`, data, uploads, artifacts, runtime files, virtualenv, node_modules, model weights and SQLite paths were confirmed ignored with `git check-ignore`.

After `de54447`, the actual checkout also passed `uv run --locked pytest -q`: **59 passed in 3.29 s, zero skips**. `uv lock --check` and `git diff --check` exited 0. The inventory CLI was rerun successfully in this checkout and recorded revision `de54447533c87bc7bfefa9e436228a913655ba29` at `2026-10-22T22:43:11.860430+00:00` (2026-10-23 in Asia/Jakarta). It observed the reference hardware, 6.037 GiB available RAM, Balanced power scheme and battery power. This snapshot is inventory, not benchmark or resource-safety evidence. The generated JSON remains in private application storage outside Git.

Inventory lock checksums describe actual file bytes; Windows checkout line-ending conversion can produce different hashes between an exported tree and the original working copy. No deterministic experiment rerun is claimed by this setup check.

The fourth commit publishes this evidence and the README delivery table. Its identifier is available from Git history after creation; the document does not attempt to embed its own commit hash. Day 1 is committed and verified; all four milestone gates remain unpassed. No remote push or release is included in this delivery.
