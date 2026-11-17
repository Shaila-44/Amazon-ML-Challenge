#!/bin/bash
cd /home/ubuntu/amazon
PY=/home/ubuntu/amazon/code/business_entity_resolution/.venv/bin/python3
while kill -0 15297 2>/dev/null; do sleep 15; done
grep -q " OK$" rescore_test.log || { echo "rescore did not finish OK -- stopping chain" > chain.log; exit 1; }
{
  echo "=== determinism check: rebuild v1 @0.65 and diff against submitted file ==="
  $PY -u make_submission.py --threshold 0.65 --out /home/ubuntu/amazon/output_v1_rebuild/matching_results.tsv --compare /home/ubuntu/amazon/output/matching_results.tsv
  echo "=== submission #2: threshold 0.96, no dedup ==="
  $PY -u make_submission.py --threshold 0.96 --out /home/ubuntu/amazon/output_v2/matching_results.tsv
} > chain.log 2>&1
$PY -u phase2b.py > phase2b.log 2>&1
$PY -u phase3_blocking.py > phase3_blocking.log 2>&1
