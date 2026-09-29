#!/bin/sh
# phonon-cpu: `serve` answers OpenAI-style requests for model "phonon-2" out of the box; everything else passes through to `fermion`.
if [ "$1" = "serve" ]; then
  case " $* " in *" --served-model-name "*) ;; *) set -- "$@" --served-model-name phonon-2;; esac
fi
exec fermion "$@"
