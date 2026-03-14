# Sync local code to Tufts HPC remote
# Usage: powershell -File scripts/sync.ps1

$LOCAL_ROOT = Split-Path -Parent $PSScriptRoot
$REMOTE = "tufts-login"
$REMOTE_ROOT = "/cluster/tufts/c26sp1cs0137/pliu07/assignment2"

Write-Host "Syncing $LOCAL_ROOT -> ${REMOTE}:${REMOTE_ROOT}" -ForegroundColor Cyan

# Create remote directory structure
ssh $REMOTE "mkdir -p $REMOTE_ROOT/{data_preparation,models,scripts,tests,inference,evaluation/baseline_cnn}"

# Sync code files (exclude data, runs, __pycache__, .git)
$files = @(
    "train.py",
    "saliency.py",
    "data_preparation/__init__.py",
    "data_preparation/dataset.py",
    "data_preparation/data_spec.py",
    "data_preparation/generate_dataset.py",
    "models/__init__.py",
    "models/cnn_baseline.py",
    "models/cnn_multi_frame.py",
    "models/cnn_3d.py",
    "models/vit.py",
    "scripts/train.slurm",
    "scripts/saliency.slurm",
    "tests/smoke_test.py",
    "inference/__init__.py",
    "inference/predict.py",
    "evaluation/baseline_cnn/model.py"
)

foreach ($f in $files) {
    $local_path = Join-Path $LOCAL_ROOT $f
    if (Test-Path $local_path) {
        $remote_path = "${REMOTE}:${REMOTE_ROOT}/${f}"
        Write-Host "  $f" -ForegroundColor Gray
        scp $local_path $remote_path
    }
}

# Fix line endings for shell scripts
ssh $REMOTE "find $REMOTE_ROOT/scripts -name '*.slurm' -exec sed -i 's/\r$//' {} \;"

Write-Host "`nSync complete!" -ForegroundColor Green
Write-Host "To submit training: ssh $REMOTE 'cd $REMOTE_ROOT; sbatch scripts/train.slurm'"
