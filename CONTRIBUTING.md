# Contributing

Issues and pull requests are welcome in Korean or English. For a change to the research method or evaluation protocol, describe the hypothesis and expected effect before running a new experiment.

## Local checks

Use Python 3.11 or newer. These checks run on CPU with synthetic fixtures and do not download the base model or AI Hub data.

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q scripts examples
```

For training dependencies and GPU setup, see [setup](docs/setup.md).

## Changes and evidence

- Keep changes focused and explain the problem, resulting behavior, and validation.
- Add a small synthetic regression check when fixing a data, ranking, or checkpoint-integrity bug.
- Preserve the original-versus-adapted retrieval comparison, frozen document-family splits, and held-out test set.
- Report the dataset version, base revision, seeds, training exposure, device, precision, and retrieval protocol for new results. Preserve negative findings.
- Use the development set for selection. Do not tune on the held-out test set.
- Commit only code, documentation, and reviewed aggregate results. Keep licensed text, model weights, tokens, remote access details, and raw execution logs outside Git.

The local regression suite checks software invariants. A green CI run does not establish retrieval quality or GPU compatibility.
