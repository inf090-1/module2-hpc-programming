#!/bin/bash
set -euo pipefail

# Build profiling target
hipcc -O2 -o profiling_target profiling_target_hip.cpp

echo "== Device info (rocminfo) =="
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | head -n 30 || true
else
  echo "rocminfo not found"
fi

echo "== rocprofv3 stats + trace (pftrace for Perfetto) =="
# We generate both:
#  - a compact perfetto-compatible pftrace timeline (for UI.perfetto.dev)
#  - json/csv stats outputs
rocprofv3 \
  --stats --kernel-trace --hip-runtime-trace --memory-copy-trace \
  --summary --summary-output-file rocprofv3_summary.txt \
  --output-directory rocprofv3_out \
  --output-file rocprofv3_out \
  -f pftrace json csv \
  -- ./profiling_target

# Also try to collect HSA trace if available (optional):
# rocprofv3 --stats --kernel-trace --hsa-trace ... -- ./profiling_target

if [ -d "rocprofv3_out" ] && ls rocprofv3_out/*results.pftrace >/dev/null 2>&1; then
  echo "[solution] rocprofv3 outputs generated (pftrace + traces) | PASS"
else
  echo "[solution] expected rocprofv3 outputs missing | FAIL"
  ls -l rocprofv3_out 2>/dev/null || true
  exit 1
fi
