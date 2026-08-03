"""Contextual help for detailed Settings pages."""
SETTINGS_HELP_ENTRIES={
"settings.paths.resolved":{"title":"Resolved Path Settings","summary":"Number of path-registry fields resolved.","details":"Includes repository, data, campaigns, MLflow, temporary, training, and publication paths."},
"settings.paths.validation":{"title":"Validated Paths","summary":"Paths confirmed by an authoritative source and filesystem checks.","details":"Validated paths agree with the repository contract or machine configuration."},
"settings.paths.attention":{"title":"Paths Needing Attention","summary":"Planned, unknown, missing, or inconsistent settings.","details":"These remain visible rather than being guessed."},
"settings.machine.current":{"title":"Current Machine","summary":"Machine identity from SCALEBRIDGE_MACHINE_ID.","details":"Controls metadata, export identity, and workload guidance."},
"settings.machine.hostname":{"title":"Hostname","summary":"Operating-system hostname.","details":"Diagnostic context and fallback only."},
"settings.machine.platform":{"title":"Detected Platform","summary":"Current operating system.","details":"Separates Windows and Linux/HPC behavior."},
"settings.environment.active":{"title":"Active Environment","summary":"Current Conda environment.","details":"Discovered from CONDA_DEFAULT_ENV."},
"settings.environment.python":{"title":"Python Version","summary":"Interpreter version running BGIRS.","details":"Should match the validated lock."},
"settings.environment.executable":{"title":"Python Executable","summary":"Absolute interpreter path.","details":"Confirms the intended environment is active."},
"settings.mlflow.connection":{"title":"MLflow Connection","summary":"Reachability of the tracking endpoint.","details":"BGIRS does not manage the server process."},
"settings.mlflow.machine":{"title":"MLflow Machine Identity","summary":"Machine metadata used for exports.","details":"Not a substitute for artifact discovery."},
"settings.mlflow.control":{"title":"MLflow Server Control","summary":"Current policy is inspect only.","details":"BGIRS validates and opens the UI but does not start or stop MLflow."},
"settings.mlflow.tracking_uri":{"title":"Tracking URI","summary":"Endpoint used by MLflow clients.","details":"Usually localhost:5000 for local development."},
"settings.mlflow.backend":{"title":"Backend Store","summary":"MLflow metadata location.","details":"Only lab-PC is currently authoritative."},
"settings.mlflow.artifacts":{"title":"Artifact Root","summary":"Logged run-artifact location.","details":"Artifacts remain outside the repo."},
"settings.mlflow.exports":{"title":"MLflow Export Root","summary":"Machine-specific metadata export location.","details":"Generated root/mlflow_exports/<machine_id>."},
"settings.mlflow.registry":{"title":"Merged Experiment Registry","summary":"Cross-machine merged metadata.","details":"Preferred for unified summaries."},
"settings.help.coverage":{"title":"Help Coverage","summary":"Total contextual-help entries.","details":"Important controls require complete help."},
"settings.help.settings_coverage":{"title":"Settings Help Coverage","summary":"Help entries dedicated to Settings.","details":"Covers paths, machines, environments, MLflow, visualization, and policy."},
"settings.help.safety":{"title":"Settings Safety Model","summary":"Inspect authoritative settings and permit low-risk changes.","details":"Scientific roots are protected; preferences are editable."}}
PATHS={"repository_root":"Repository Root","data_root":"Primary Data Root","generated_data_root":"Generated Data Root","campaigns_root":"Campaigns Root","generation_template":"Generation Template","aggregation_template":"Aggregation Template","phase_c_template":"Phase C Template","phase_d_template":"Phase D Template","mlflow_backend_root":"MLflow Backend Root","mlflow_artifact_root":"MLflow Artifact Root","mlflow_export_root":"MLflow Export Root","experiment_registry_root":"Experiment Registry","energyplus_work_root":"EnergyPlus Work Root","temporary_root":"Temporary Root","publication_export_root":"Publication Export Root","training_export_root":"Training Export Root"}
for key,title in PATHS.items(): SETTINGS_HELP_ENTRIES[f"settings.path.{key}"]={"title":title,"summary":f"Resolved {title.lower()} and provenance.","details":"Shows value, source, status, and filesystem checks. Scientific roots are not freely editable."}
