# ScaleBridge Week 0 Status

Date: 2026-05-31  
Week: Week 0  
Role: Foundation freeze  

## Week 0 Purpose

Week 0 restarts and formalizes the software-development side of the PhD research workflow.

The goal is to prepare a stable foundation for Week 1 coding, not to start deep model implementation.

## Completed

- Coding planner created in Google Sheets.
- Repository name selected: `scalebridge-research`.
- Python package name selected: `scalebridge`.
- Top-level repository folders created.
- Package folder skeleton created under `src/scalebridge/`.
- Paper-specific experiment folders created.
- Initial README, pyproject, gitignore, and environment example created.
- Legacy reference policy documented.
- Package architecture documented.

## Week 0 Validation Result

The repository structure exists and passed initial inspection.

The following required files were created during final Week 0 closure:

- `docs/migration/legacy_source_policy.md`
- `docs/architecture/package_design.md`
- `docs/development_notes/week0_status.md`
- `scalebridge.yaml`
- `scalebridge_cpu.yaml`
- `check_scalebridge_environment.py`
- `scalebridge_environment_notes.md`

## Not Yet Fully Validated

The following must still be checked:

- editable package install using `pip install -e .`,
- `import scalebridge`,
- full environment creation or update,
- dependency smoke test,
- MLflow import,
- PyTorch import,
- Neuromancer import,
- Optuna/Ray imports,
- EnergyPlus/OpenDSS-related imports.

## Machine Status

| Machine | Status |
|---|---|
| dev-laptop | Repository structure inspected; environment still needs smoke test |
| home-pc | Not tested |
| lab-pc | Not tested |
| wsu-hpc | Not tested |

## Week 1 Entry Condition

Week 1 may start after:

1. missing Week 0 files are created,
2. editable package import is tested,
3. environment smoke test is attempted,
4. failures are documented honestly.

## Week 1 Focus

Week 1 should focus on package foundation and P1 data-pipeline skeleton:

- core path utilities,
- config loading,
- P1 dataset manifest format,
- P1 preprocessing skeleton,
- PyTorch Dataset/DataLoader skeleton,
- basic smoke tests,
- MLflow run skeleton.

Do not begin full P1 model experiments until the data pipeline and package import are stable.
