# Contributing to Cosmea

[日本語](CONTRIBUTING.ja.md)

## Project settings

| Setting | Decision / source of truth |
| --- | --- |
| Deliverable | Python retrieval-training library, `cosmea` |
| Environment | CPU, single training process, finite map-style datasets; locally verified on macOS arm64; Ubuntu 24.04 runs the same checks in [GitHub Actions](https://github.com/tkhshtsh0917/cosmea/actions/workflows/ci.yml) |
| Runtime | Development Python is pinned in `.python-version`; supported minimum is declared in `pyproject.toml` |
| Tools | mise manages uv and Betterleaks versions in `mise.toml`; uv manages Python and the virtual environment |
| Dependencies | Compatibility ranges and extras in `pyproject.toml`; exact development resolution in committed `uv.lock` |
| Formatting / lint / types | Ruff format and lint; mypy for all package modules. Configurations in `pyproject.toml` |
| Verification | `mise run check`: format, lint, types, tests, offline training examples, build, public-content audit and Betterleaks |
| Compatibility | Preserve exported class names and metadata ordering. Changes to proposal measures, masks or defaults require corresponding contract updates |
| Resource limits | `0 <= K <= M`, `1 <= B <= N`, positive cached chunk size. Detailed compute and memory scope in [design](docs/design.md) |
| Public documentation | English README, API, design and contributor guide, with Japanese counterparts |
| Distribution | Wheel and sdist built locally. No registry, deployment or public release configured; project license is not yet assigned, see [NOTICE](NOTICE.md) |
| Acceptance | Checks pass; public docs are self-contained; artifacts exclude private records. No claim of real-data performance |
| Git permissions | Local edits/tests are within the assigned development task. Commit, history rewrite, push, tag and publication each need explicit maintainer authorization; none is implied by a passing check |

The package version is defined by `project.version`; the exported `__version__` must agree. Runtime dependencies are Sentence Transformers, PyTorch and NumPy. The `train` extra supplies Trainer integration dependencies (datasets/Accelerate); `dev` supplies development tools. Sentence Transformers uses Apache-2.0. PyTorch and NumPy distributions include multiple licenses; consult the installed metadata and bundled LICENSE files. These are separate from the license of Cosmea itself. Updating the pinned ST minor version requires rechecking the collator, Trainer metadata transfer and GradCache interfaces.

## Setup and validation

Install [mise](https://mise.jdx.dev/getting-started.html), inspect the checkout configuration, then run:

```bash
mise trust
mise install
mise run setup
mise run check
```

`setup` is the explicit dependency-install step. Check tasks use `uv run --frozen --no-sync` and build without isolation so they do not silently change tool or dependency versions. A fresh environment must run setup first. For intentional updates, change the relevant source of truth, run `mise exec -- uv lock`, review the lock diff, then setup and check again. `mise run format` applies formatting; `format-check` only verifies it.

All tests use local synthetic inputs. Statistical tests are small enumeration/Monte Carlo checks, not production benchmarks. The worker test needs OS shared memory. In constrained environments, run both parts separately and state any unexecuted part:

```bash
mise exec -- uv run --frozen --no-sync pytest -m 'not workers' -q
mise exec -- uv run --frozen --no-sync pytest -m workers -q
```

The CI workflow runs the same setup and check tasks with read-only repository permission. Creating a workflow does not constitute a hosted CI run. No publishing job is included.

## Changes and reviews

Separate responsibilities across `src/cosmea`, `tests`, `examples`, `experiments` and `scripts`. Keep meaningful blank lines between input validation, computation and state changes; apply the same readability standard to tests and tooling. Explain input shapes, failures, numerical choices and side effects in docstrings. Put attribution and development history in human documentation.

Fixes need behavioral regression coverage and checks of related boundary inputs and call paths. Test against independent mathematical oracles where possible. Do not replace a proof with a stationarity check or treat a synthetic training run as effectiveness evidence. Review code, dependencies, diagrams, documentation and packaging together. For performance claims, record both versions, inputs, environment, warmup, repetition and aggregation, including regressions on other inputs.

Maintain English and Japanese docs together. Keep public files free of personal paths, private references and secrets. `scripts/check_public.py` checks navigation, obvious local paths and artifact payloads; it is not a general secret scanner. Preserve upstream attribution. Keep private records out of Git and artifacts, and preserve historical snapshots unchanged.

## Builds and future releases

`mise run build` produces one wheel and sdist in `dist`; `mise run audit` inspects both. Remove stale build artifacts when changing versions. Third parties can build from the sdist without private inputs. Before a future release, a maintainer must choose a project license and release destination, authorize the Git/release actions, confirm hosted CI, inspect public history/author identity, and record artifact checksums. Publishing is outside this development task.

## Secret scanning and readable examples

`mise run secrets` runs [Betterleaks](https://github.com/betterleaks/betterleaks/tree/v1.10.1) (MIT), pinned in `mise.toml`, after the build. It scans the public file inventory, unpacked wheel/sdist contents, and all locally available Git refs when commits exist. CI checks out full history; shallow local clones must fetch missing history. An uncommitted repository or source archive reports the absence of history explicitly. Detection uses the pinned built-in rules, with provider validation disabled; findings and scanner errors fail the check. Redacted JSON reports stay in ignored `build/security/`. No blanket allowlist or baseline is applied. This is pattern detection, not a guarantee that all secrets are absent.

Format multiline calls with one argument per line and a trailing comma. Apply this to public code examples as well as source code; keep short, simple calls on one line when that is clearer. Split grouped imports when a long import would impede reading.
