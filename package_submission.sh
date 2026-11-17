#!/usr/bin/env bash
# Assembles the final submission zip exactly as problem_statement.md specifies:
#   <team_name>_submission.zip
#   ├── output/matching_results.tsv
#   ├── output/candidate_pairs.tsv
#   ├── code/business_entity_resolution/{src/,README.md,requirements.txt,notebooks/}
#   └── Documentation_template.md
#
# Run from the project root (where this script lives) after output/ is populated
# by a real pipeline run and Documentation_template.md is filled in.
set -euo pipefail

TEAM_NAME="Solaris"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$ROOT/.submission_stage"
ZIP_NAME="${TEAM_NAME}_submission.zip"

for f in output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md; do
  if [ ! -s "$ROOT/$f" ]; then
    echo "Missing or empty: $f -- run the pipeline and fill in the doc before packaging." >&2
    exit 1
  fi
done

rm -rf "$STAGE" "$ROOT/$ZIP_NAME"
mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"

cp "$ROOT/output/matching_results.tsv" "$STAGE/output/"
cp "$ROOT/output/candidate_pairs.tsv" "$STAGE/output/"
cp "$ROOT/Documentation_template.md" "$STAGE/"

rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.ipynb_checkpoints' \
  "$ROOT/code/business_entity_resolution/" "$STAGE/code/business_entity_resolution/"

cd "$STAGE"
zip -rq "$ROOT/$ZIP_NAME" .
cd "$ROOT"
rm -rf "$STAGE"

echo "Wrote $ZIP_NAME"
unzip -l "$ZIP_NAME"
