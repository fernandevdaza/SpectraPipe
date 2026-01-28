# Contributing to SpectraPipe

Thank you for your interest in contributing! SpectraPipe is an open-source tool designed for the scientific community, and we value collaboration.

## How Can I Help?

We are currently focused on the following areas:

1.  **Model Modularity:** Our main goal is to decouple the MST++ model from the CLI core. We are looking to refactor the code to allow new Deep Learning models to be "plug-and-play".
2.  **Format Support:** Add readers for new hyperspectral image formats (ENVI, TIFF).
3.  **Documentation:** Improvements to docstrings and usage tutorials.

## Workflow

1.  Fork the repository.
2.  Create a branch for your feature (`git checkout -b feature/new-modularity`).
3.  Ensure your code meets style standards (we use `ruff`/`flake8` and strict typing).
4.  Add unit tests for new functionality (especially for mathematical calculations or metrics).
    *   Run `pytest` to ensure no regressions.
5.  Submit a Pull Request describing your changes.

## Code Standards

*   **Typing:** All code must have Python Type Hints (`typing`).
*   **Reproducibility:** Any change affecting image processing must be deterministic or allow for a fixed random seed.
*   **Schema:** If you modify the output `.npz` structure, you MUST update the schema documentation in `DATA_CONTRACTS.md`.

## Bug Reporting

Please include:
*   The exact command you ran.
*   A sample `manifest.yaml` or input file (where applicable).
*   Full error logs (stack trace).
*   Pipeline version (commit hash, or similar).

---
**CI Note:**
The system includes mocks for execution without a GPU. Ensure your tests pass locally using `poetry run pytest`.
