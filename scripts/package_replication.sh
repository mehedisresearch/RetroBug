#!/usr/bin/env bash
# Build a clean RetroBug replication archive (RQ1–RQ3 only; no ablation study).
#
# Usage (from BugFix/):
#   bash scripts/package_replication.sh
#   bash scripts/package_replication.sh /path/to/output-dir
#
# Produces: retrobug-replication-YYYYMMDD.tar.gz

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d)"
OUT_DIR="${1:-${ROOT}/dist}"
STAGE="${OUT_DIR}/retrobug-replication-${STAMP}"
ARCHIVE="${OUT_DIR}/retrobug-replication-${STAMP}.tar.gz"

echo "==> Staging replication package at ${STAGE}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}"

# --- agentic_experiment (core package, no ablation) ---
mkdir -p "${STAGE}/agentic_experiment"

rsync -a \
  --exclude='.env' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='data/ablation/' \
  --exclude='ablation/' \
  --exclude='run_ablation.py' \
  --exclude='summarize_ablation.py' \
  --exclude='paper_tables/' \
  --exclude='data/cbr_rag/backward/' \
  "${ROOT}/agentic_experiment/" "${STAGE}/agentic_experiment/"

# --- source datasets ---
mkdir -p "${STAGE}/bugfix_pipeline/data"
cp "${ROOT}/bugfix_pipeline/data/dataset1_100fix_300nonfix.csv" "${STAGE}/bugfix_pipeline/data/"
cp "${ROOT}/bugfix_pipeline/data/dataset2_100fix_300nonfix.csv" "${STAGE}/bugfix_pipeline/data/"

# --- shared LLM baseline helpers ---
mkdir -p "${STAGE}/claude_message_experiment"
cp "${ROOT}/claude_message_experiment/prompts.py" "${STAGE}/claude_message_experiment/"
cp "${ROOT}/claude_message_experiment/parse_response.py" "${STAGE}/claude_message_experiment/"

cp "${ROOT}/README.md" "${STAGE}/"

# --- manifest ---
cat > "${STAGE}/MANIFEST.txt" <<EOF
RetroBug replication package (${STAMP})
Scope: RQ1 (non-agentic baselines), RQ2 (Forward CBR vs RetroBug), RQ3 (cross-fold stability)
Excluded: ablation study, data/ablation/, data/cbr_rag/backward/, paper_tables/

Entry point: README.md

Included:
  agentic_experiment/     main code + precomputed results
  bugfix_pipeline/data/   source CSV datasets (798 commits)
  claude_message_experiment/  shared LLM prompt + parser
EOF

mkdir -p "${OUT_DIR}"
tar -czf "${ARCHIVE}" -C "${OUT_DIR}" "retrobug-replication-${STAMP}"

SIZE="$(du -sh "${ARCHIVE}" | cut -f1)"
echo "==> Created ${ARCHIVE} (${SIZE})"
echo "==> Unpack with: tar -xzf ${ARCHIVE}"
