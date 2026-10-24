# Frozen line purpose and quality pilot

This local clean checkout contains the selected V6 zero-shot four-model workflow, separate evaluation and Gold association analysis. It is prepared for review, not published to GitHub. No model training or Gold purpose accuracy is claimed. See Research_report.md in the parent deliverables for findings and limitations.

## Results

LLTC4J exposed development: 20 hunks / 171 lines, consensus accuracy60.8%, macro-F158.2%. Gold:70 fresh hunks /242 changed lines, each model242 accepted predictions, consensus240. Gold supplies original hunk quality labels only. All four are Qwen variants and refactoring performance is poor. Original raw outputs are private external dependencies, not synthesized or bundled.

## Setup

Use Python matching ENVIRONMENT.md. Create an isolated environment and install compatible GPU dependencies from requirements-hpc.txt after checking the site's CUDA/driver. Read pinned models.json and official_assets.json. Obtain weights directly from the official pinned model revisions under their respective licenses. Keep weights in one shared cache. No paid services are needed for the recorded free student allocation. Slurm templates are site-specific UNC Charlotte examples and contain the original user/project paths; adapting them to another machine creates a new recorded environment.

## Required authorized data

Restore LLTC4J prepared inputs.jsonl and references.jsonl into data/prepared_v6_dev using the exact hashes in its manifest and evaluation_manifest. Restore original workbook at ../Datasets/Our_Dataset/projects_tangled_code.xlsx; original hash is in the Gold audit. Source datasets are excluded from this checkout and must be acquired with appropriate permission. For exact final Gold replay restore inputs.jsonl and review_metadata.jsonl to data/prepared_gold_final_20261024 and the prior pilot metadata to data/prepared_gold_v6_pilot_20261023; hashes and source row IDs remain in project manifests. Preparation script also requires the clean development run's environment/metrics and excludes the prior Gold pilot commits. Run preparation only in a fresh destination; do not overwrite prepared data or frozen baselines.

## GPU inference

Before submission, verify free allocation, disk capacity, model assets and input hashes. On the recorded HPC project root:

```bash
sbatch --export=ALL,LLTC4J_NO_ADDITIONAL_CHARGE_VERIFIED=yes hpc/v6_clean_verified_20261024/job.sbatch
sbatch --export=ALL,LLTC4J_NO_ADDITIONAL_CHARGE_VERIFIED=yes hpc/gold_final_20261024/job_pathfix.sbatch
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

Submit sequentially as needed, not both unnecessarily. Never run heavy inference on the login node. Templates create fresh job-specific output directories and use the original immutable V6 inference source. stage.py intentionally requires /tmp/lltc4j-v6-clean-JOBID; do not change this path without updating the protocol. Only inputs and code are sent to the predictor; references and original Gold quality metadata stay local.

## Separate local analysis

After downloading and verifying the raw archive, restore its tree under hpc/records/. Preserve raw responses, per-hunk native IDs and native-stop/error flags.

```bash
python hpc/v6_clean_verified_20261024/test_contract.py
python hpc/v6_clean_verified_20261024/score.py --run hpc/records/v6-clean-27460354/runs/v6-clean-27460354 --output reports/new_lltc_evaluation
python hpc/gold_final_20261024/analyze.py --job 27461174
```

The Gold analyzer refuses to overwrite reports/gold_final_JOBID. Reference aggregate results are kept under reference_results/ so the active output path is empty. It verifies input/prompt/model/source hashes, all model completion markers and exact line-ID alignment, then writes consensus, line comparison, hunk summaries, coverage, pairwise agreement, relation estimates, sampling weights and cluster-bootstrap intervals. No raw file is edited. LLTC scoring currently derives ROOT from its source location; keep paths unchanged. Aggregate stored metrics are reference artifacts, not newly executed results.

## Sampling and analysis

Gold SHA256 ranking seed gold-final-v6-20261024; minimum five per quality with proportional allocation; prior pilot commits, large context and exact duplicate diffs excluded. Inclusion probabilities are retained. Conditional accepted-line proportions and all-changed/hunk-weighted alternatives are separate. Overall ratios use inverse inclusion weights. Bootstrap uses2000 paired project+commit cluster draws within quality and percentile intervals, seed20261024. Do not treat lines as independent, ignore abstentions, infer causality/entanglement, or call agreement accuracy.

## Reproducibility scope

The clean development run repeated every model twice and matched tokens/text/labels exactly on the same environment. Gold ran once. Other hardware/frameworks may differ. No Ollama export, quantized local deployment, adapters or Mac latency evaluation is included. Workbooks require the Codex artifact runtime as documented under artifacts/. Synthetic example under examples/ shows schema only, not real validation data. See source_manifest.json for this checkout's file hashes.
