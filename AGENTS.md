# Cosmea development rules

- Read `CONTRIBUTING.md` for project settings, version ownership and verification commands.
- Keep package code, examples, tests, experiments and development scripts separate. Training-only dependencies belong to the `train` extra; development tools belong to the `dev` dependency group.
- Match every proposal correction to the actual sampling, masking and deduplication design. Do not equate unbiased partition estimates with unbiased loss gradients or finite-sample consistency.
- Explain responsibilities, shapes, failure conditions and numerical constraints in docstrings. Keep attribution and development history in documentation, preserving required copyright notices.
- Update English and Japanese public documentation together. Public documentation must be sufficient without personal notes, private files or machine-specific paths.
- Preserve historical evidence bytes and hashes. Keep private work records outside version control and distributions.
- Use `mise run setup` after an intentional dependency change; use `mise run check` for validation. Checks must not resolve or update dependencies.
- Add behavior-based regression tests for fixes and examine related input boundaries. Do not suppress checks globally to hide failures.
- Keep source-only edits, commit creation, history rewriting, push, tags and publication as distinct authorizations. Preserve other contributors' work.
- Report actual commands and results. CI configuration is not evidence that hosted CI ran; synthetic smoke tests are not evidence of real-data quality.
- Use one argument per line in multiline calls, including documentation examples; keep simple short calls compact. Run the pinned Betterleaks scan as part of the shared checks.
