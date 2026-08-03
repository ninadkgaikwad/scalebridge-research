Machine-local overrides may be stored here as:

    <machine_id>.local.json

These files may contain machine-specific absolute paths. Add the following rule
to the repository .gitignore if it is not already present:

    config/local/*.local.json
