#!/usr/bin/env bash
set -euo pipefail

# Runs rocprof-compute roofline (roof-only) for:
#   1) HIP matmul
#   2) OpenMP target matmul
#   3) rocBLAS matmul
# and generates PDF roofline plots under a working directory.
#
# Optional: if ImageMagick 'convert' exists on the node, it will also create PNGs.
#
# Expected executables (same directory as this script):
#   - ./hip_matmul
#   - ./omp_target_matmul
#   - ./rocblas_matmul

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="${1:-roofline_matmul_omp_vs_hip}"
mkdir -p "$OUT_DIR"

HIP_EXE="${HIP_EXE:-./hip_matmul}"
OMP_EXE="${OMP_EXE:-./omp_target_matmul}"
ROC_EXE="${ROC_EXE:-./rocblas_matmul}"

if [[ ! -x "$HIP_EXE" ]]; then
  echo "[ERROR] HIP executable not found/executable: $HIP_EXE" >&2
  exit 1
fi
if [[ ! -x "$OMP_EXE" ]]; then
  echo "[ERROR] OpenMP target executable not found/executable: $OMP_EXE" >&2
  exit 1
fi
if [[ ! -x "$ROC_EXE" ]]; then
  echo "[ERROR] rocBLAS executable not found/executable: $ROC_EXE" >&2
  exit 1
fi

HIP_WORKDIR="$OUT_DIR/hip"
OMP_WORKDIR="$OUT_DIR/omp_target"
ROC_WORKDIR="$OUT_DIR/rocblas"
mkdir -p "$HIP_WORKDIR" "$OMP_WORKDIR" "$ROC_WORKDIR"

ROOFLINE_TYPE="${ROOFLINE_TYPE:-FP32}"
DEVICE_ID="${DEVICE_ID:-0}"
DIM_ARG="${DIM_ARG:-0}"
# If DIM_ARG > 0, pass it as first positional argument to each executable.

export ROCPROFILER_SDK_LIBRARY_PATH="${ROCPROFILER_SDK_LIBRARY_PATH:-}"

LIBOMP_DIR=""
if [[ -d "/opt/rocm-7.2.0/lib/llvm/lib" ]]; then
  LIBOMP_DIR="/opt/rocm-7.2.0/lib/llvm/lib"
elif [[ -d "/opt/rocm-7.2.3/lib/llvm/lib" ]]; then
  LIBOMP_DIR="/opt/rocm-7.2.3/lib/llvm/lib"
fi

if [[ -n "$LIBOMP_DIR" ]]; then
  export LD_LIBRARY_PATH="$LIBOMP_DIR:${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$LIBOMP_DIR/libomp.so"
fi

HIP_CMD=("$HIP_EXE")
OMP_CMD=("$OMP_EXE")
ROC_CMD=("$ROC_EXE")
if [[ "$DIM_ARG" != "0" ]]; then
  HIP_CMD+=("$DIM_ARG")
  OMP_CMD+=("$DIM_ARG")
  ROC_CMD+=("$DIM_ARG")
fi

echo "[INFO] Running rocprof-compute roofline (HIP) ..."
/opt/rocm/bin/rocprof-compute profile \
  --roof-only \
  -R "$ROOFLINE_TYPE" \
  --name "matmul_hip_roof" \
  --path "$HIP_WORKDIR" \
  --device "$DEVICE_ID" \
  --format-rocprof-output csv \
  -- env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" LD_PRELOAD="$LD_PRELOAD" "${HIP_CMD[@]}"

echo "[INFO] Running rocprof-compute roofline (OpenMP target) ..."
/opt/rocm/bin/rocprof-compute profile \
  --roof-only \
  -R "$ROOFLINE_TYPE" \
  --name "matmul_omp_target_roof" \
  --path "$OMP_WORKDIR" \
  --device "$DEVICE_ID" \
  --format-rocprof-output csv \
  -- env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" LD_PRELOAD="$LD_PRELOAD" "${OMP_CMD[@]}"

echo "[INFO] Running rocprof-compute roofline (rocBLAS) ..."
/opt/rocm/bin/rocprof-compute profile \
  --roof-only \
  -R "$ROOFLINE_TYPE" \
  --name "matmul_rocblas_roof" \
  --path "$ROC_WORKDIR" \
  --device "$DEVICE_ID" \
  --format-rocprof-output csv \
  -- env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" LD_PRELOAD="$LD_PRELOAD" "${ROC_CMD[@]}"

# Extract roofline PDFs.
find_roof_pdf() {
  local root="$1"
  local pattern="${2:-*empirRoof*.pdf}"
  find "$root" -type f -iname "$pattern" | head -n 1 || true
}

HIP_PDF="$(find_roof_pdf "$HIP_WORKDIR" "*empirRoof*.pdf")"
OMP_PDF="$(find_roof_pdf "$OMP_WORKDIR" "*empirRoof*.pdf")"
ROC_PDF="$(find_roof_pdf "$ROC_WORKDIR" "*empirRoof*.pdf")"

if [[ -z "$HIP_PDF" || -z "$OMP_PDF" || -z "$ROC_PDF" ]]; then
  echo "[ERROR] Could not find one or more roofline PDFs." >&2
  echo "[ERROR] HIP: $HIP_PDF" >&2
  echo "[ERROR] OMP: $OMP_PDF" >&2
  echo "[ERROR] ROC: $ROC_PDF" >&2
  exit 4
fi

echo "[INFO] HIP roofline PDF: $HIP_PDF"
echo "[INFO] OMP roofline PDF: $OMP_PDF"
echo "[INFO] ROCBLAS roofline PDF: $ROC_PDF"

HIP_PNG="$OUT_DIR/hip_roofline.png"
OMP_PNG="$OUT_DIR/omp_target_roofline.png"
ROC_PNG="$OUT_DIR/rocblas_roofline.png"
COMBINED_PNG="$OUT_DIR/roofline_matmul_3way.png"

if command -v convert >/dev/null 2>&1; then
  convert -density 200 "$HIP_PDF[0]" "$HIP_PNG" 2>/dev/null || convert -density 200 "$HIP_PDF" "$HIP_PNG"
  convert -density 200 "$OMP_PDF[0]" "$OMP_PNG" 2>/dev/null || convert -density 200 "$OMP_PDF" "$OMP_PNG"
  convert -density 200 "$ROC_PDF[0]" "$ROC_PNG" 2>/dev/null || convert -density 200 "$ROC_PDF" "$ROC_PNG"
  convert "$HIP_PNG" "$OMP_PNG" "$ROC_PNG" +append "$COMBINED_PNG"
  echo "[INFO] Generated: $COMBINED_PNG"
else
  echo "[WARN] 'convert' not found on this node. PDFs only were generated."
fi

echo "[INFO] Raw outputs stored in: $OUT_DIR"
