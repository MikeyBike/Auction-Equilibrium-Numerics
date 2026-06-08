# Auction-Equilibrium-Numerics

`Auction-Equilibrium-Numerics` is a JAX-based Python package scaffold for auction
equilibrium numerics.

This repository currently provides project infrastructure only:

- `uv`-managed Python environment and lockfile
- `src/`-layout package with type information
- linting, type checking, testing, and pre-commit hooks
- GitHub Actions CI and basic issue/PR templates

The numerical methods and domain-specific package structure will be added later.

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run mypy src
uv run pytest
uv build
```
