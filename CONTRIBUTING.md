<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to Alpamayo 2 Edge

Contributions can improve the inference code, examples, notebooks, tests, and documentation.

## Report an issue

Use the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766)
for usage questions. Open a [GitHub issue](https://github.com/NVlabs/alpamayo2-edge/issues)
for code bugs, documentation issues, or feature requests.

For a bug report, include the command, expected result, actual result, and steps to reproduce it.
Include relevant package versions and GPU details. Use public inputs that others can access.
Report security vulnerabilities through [SECURITY.md](SECURITY.md).

## Development setup

Use Python 3.12 and `uv`. See the [prerequisites](README.md#prerequisites) for inference requirements.
Run these commands from the repository root:

```bash
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --locked --dev
source .venv/bin/activate
```

For notebook changes, install the `notebook` extra as described in
[Notebook inference](README.md#notebook-inference). For acceleration changes, see
the [acceleration requirements](docs/acceleration.md#requirements).

## Checks

Follow the code style in the files that you change. Add or update tests when a change affects behavior.
Run the tests that cover the change. To run the full test suite:

```bash
python -m pytest
```

See [Tests](README.md#tests) for optional checkpoint and GPU tests.
Check Python code with Ruff. Check the diff for whitespace errors:

```bash
ruff check src tests
git diff --check
```

For documentation changes, check links and command examples.
Clear notebook outputs and execution counts before submitting notebook changes.

## Pull requests

Keep each pull request focused on one change. Open it against `main`.
Update the documentation and examples when the change affects their use.
Include the following information in the pull request description:

- The problem and the resulting behavior.
- The commands used to check the change and their results.
- Any model, dataset, GPU, or package requirements needed to reproduce the results.
- Any checks that you could not run.

Keep model weights, dataset files, caches, and generated inference results out of the pull request.
Do not include credentials, private paths, private URLs, or non-public data.
