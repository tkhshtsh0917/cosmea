# Contributing to Cosmea

[日本語](CONTRIBUTING.ja.md)

## Project settings

| Setting | Source of truth |
| --- | --- |
| Library | `src/cosmea`; CPU, one training process, finite datasets |
| Python | Development version in `.python-version`; supported minimum in `pyproject.toml` |
| Tools | mise pins uv and Betterleaks in `mise.toml` |
| Dependencies | Runtime ranges and the `train` extra in `pyproject.toml`; development tools in the `dev` dependency group |
| Reproducibility | `uv.lock` pins the main development and CI environment |
| Quality | Ruff formatting/lint, mypy, pytest, build, public-content and license audit, Betterleaks |
| Documentation | English and Japanese README, API, design, examples and this guide |
| License | Apache-2.0; LICENSE and NOTICE are included in wheel and sdist |

The package version is defined by `project.version`; the exported `__version__` must agree. Changes to metadata ordering, sampling, masks or defaults require matching API documentation and behavioral tests. The sampling and resource limits are explained in [design](docs/design.md).

## Setup and validation

Install [mise](https://mise.jdx.dev/getting-started.html), inspect the checkout configuration, then run:

```bash
git clone https://github.com/tkhshtsh0917/cosmea.git
cd cosmea

mise trust
mise install
mise run setup
mise run check
```

`setup` installs the locked training dependencies and `dev` group. `check` does not resolve or update dependencies: it uses `uv run --frozen --no-sync` and builds without isolation. For an intentional dependency change, edit `pyproject.toml`, run `mise exec -- uv lock`, review the diff, then run setup and check. `mise run format` applies formatting.

The main CI job runs the same locked setup/check tasks. Separate compatibility jobs explicitly resolve Python 3.11 with Torch 2.6/NumPy 2.0 and Python 3.13 within the declared ranges, record installed versions, and run tests. These jobs do not replace the locked reference environment. Actions are pinned to commit SHAs; update the SHA and version comment together.

## Tests and examples

Tests at the root of `tests/` cover the library, including small exact sampling oracles and Trainer integration. `tests/tooling/` checks development audits; `tests/experiments/` checks synthetic metric definitions. All are included in `mise run check`.

The real DataLoader worker test exercises process initialization and needs OS shared memory. If the environment restricts it, run the two groups separately and report any unexecuted group:

```bash
mise exec -- uv run --frozen --no-sync pytest -m 'not workers' -q
mise exec -- uv run --frozen --no-sync pytest -m workers -q
```

## Offline example and optional experiment

`mise run smoke` runs the ordinary and cached example with a tiny local BERT and synthetic data. It requires no model download. It is optional: the standard check already tests training through the Trainer, including mined negatives and a singleton final batch.

For a longer synthetic quality comparison:

```bash
mise exec -- uv run --frozen --no-sync python experiments/toy_two_tower.py --seeds 3 --out results.json
```

For statistical illustrations outside the regression suite:

```bash
mise exec -- uv run --frozen --no-sync python experiments/diagnostics/estimators.py --trials 2000
mise exec -- uv run --frozen --no-sync python experiments/diagnostics/stationarity.py --max-negatives 2
```

These illustrate selected distributions, not general variance rankings or proofs of consistency. Exact enumeration grows exponentially. Keep arguments, seeds, source revision and environment with results. The full default quality experiment has not been rerun for this revision. See [design](docs/design.md) for metrics and assumptions.

## Reviews and readable code

Keep responsibilities separate across `src/cosmea`, `examples`, `tests`, `experiments` and `scripts`. Use meaningful blank lines between input validation, computation and state changes. In multiline calls, put one argument per line with trailing commas; keep simple short calls compact. Explain shapes, failures and important constraints in docstrings.

Fixes need tests that would fail before the fix, preferably with independent mathematical oracles. Cover relevant boundaries without repeating the same property across every parameter combination. Synthetic interface tests do not establish real-data quality. Record the inputs, versions and measurement method for performance claims.

Update English and Japanese docs together. Public files must be sufficient without local notes and must not contain personal paths or private links. Agent-specific operation rules are in [AGENTS.md](AGENTS.md).

## Builds, licensing and secret scanning

`mise run build` produces one wheel and sdist in `dist`. The `secrets` task first performs the public-content/license audit, then runs Betterleaks; `check` invokes this combined task once. `mise run audit` remains available to run the audit alone.

The audit checks license metadata and byte-for-byte LICENSE/NOTICE contents, the typing marker, and that sdist members belong to the public source inventory or named build-generated metadata. This is separate from secret detection. Setuptools 77.0.3 or later is required for PEP 639 metadata; its development version is locked. Remove stale artifacts when changing package versions.

Betterleaks uses the pinned built-in rules and scans public files, extracted artifacts and all locally available Git refs. Provider validation is disabled; findings and scanner errors fail the task. Redacted reports remain in ignored `build/security/`. Shallow clones fail the history check: fetch the missing history yourself before retrying. Pattern detection does not guarantee that no secrets exist.

Dependencies retain their own licenses. ST uses Apache-2.0; PyTorch and NumPy distributions include multiple licenses. Preserve required third-party notices if incorporating their material. Before a package release, confirm hosted CI, inspect artifacts and licensing, record checksums, and follow the maintainer's release process. No package publishing job is configured.
