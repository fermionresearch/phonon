#!/bin/zsh
set -euo pipefail
HERE="${0:A:h}"
PYTHON="${PYTHON:-python3}"

if [[ "$(uname -m)" != "arm64" ]]; then
  print -u2 "This runtime is optimized for Apple Silicon (arm64)."
  exit 1
fi

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  "$PYTHON" -m venv "$HERE/.venv"
fi

"$HERE/.venv/bin/python" -m pip install --upgrade pip
"$HERE/.venv/bin/python" -m pip install -r "$HERE/requirements-runtime.txt"
chmod +x "$HERE/run.sh" "$HERE/stt.py" "$HERE/verify_install.py"

# The model is a separate download, so setup can legitimately run before one is
# installed.  Verify whichever models are present; if none are, say what to do
# next rather than failing.
found=0
for pair in \
  "parity:model_v18_mlx_quint5" \
  "audio6:model_v18_mlx_head8audio6_quint5" \
  "micro:model_v18_mlx_hybrid4_quint5"; do
  name="${pair%%:*}"
  dir="${pair##*:}"
  if [[ -f "$HERE/$dir/packed_manifest.json" ]]; then
    "$HERE/.venv/bin/python" "$HERE/verify_install.py" --profile "$name"
    found=1
  fi
done

if (( found )); then
  print "Ready. Run: $HERE/run.sh mic"
else
  print "Environment ready. No model is installed yet."
  print ""
  print "Pick one and follow the quickstart in README.md:"
  print "  Phonon-1 Big        581 MB  https://huggingface.co/FermionResearch/Phonon-1-Big"
  print "  Phonon-1  415 MB  https://huggingface.co/FermionResearch/Phonon-1"
  print "  Phonon-1 Micro  285 MB  https://huggingface.co/FermionResearch/Phonon-1-Micro"
  print ""
  print "For the default, that is:"
  print "  $HERE/.venv/bin/python $HERE/package_release_bps.py unpack \\"
  print "      phonon-parity.bps.tar.zst $HERE/model_v18_mlx_quint5"
  print "  $HERE/run.sh mic"
fi
