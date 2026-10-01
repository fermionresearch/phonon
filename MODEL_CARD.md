# Model cards

Each Phonon model carries its own model card on Hugging Face. The cards are maintained there, so this file points
at them rather than keeping a copy that could fall out of date. The current model is **Phonon-2** (`phonon-2`, a 164 MB
download, CC-BY-4.0): <https://huggingface.co/FermionResearch/Phonon-2>. The Phonon-1 family:

| Model | Download | On disk | Full LibriSpeech clean / other | Macro WER (8 benchmarks) | Card |
|---|---:|---:|:---:|---:|---|
| **Phonon-1** — the Phonon-1 flagship. Across five real-world benchmarks — AMI (meetings), Earnings-22 (earnings calls), GigaSpeech (web video), SPGISpeech (financial speech), TED-LIUM (talks) — no downloadable model we could find is both smaller (download bytes) and more accurate, on any of the five. | 415.1 MB | 0.455 GB | 2.640 % / 5.699 % | 7.671 | <https://huggingface.co/FermionResearch/Phonon-1> |
| **Phonon-1 Big** — the largest build. | 580.9 MB | 0.822 GB | 2.667 % / 5.722 % | 7.604 | <https://huggingface.co/FermionResearch/Phonon-1-Big> |
| **Phonon-1 Micro** — the smallest install; beats Moonshine base on all eight benchmarks. | 285.1 MB | 0.331 GB | 3.002 % / 6.511 % | 8.522 | <https://huggingface.co/FermionResearch/Phonon-1-Micro> |

LibriSpeech figures are the full test set (5,559 utterances); the macro spans
eight public benchmarks — one protocol, standard normalized WER. Each card
carries that model's full evaluation, licence and attribution.

## In this repo instead

- [README](README.md#models) — every model with its download size and weights.
- [README](README.md#run) — install and the exact commands.
- [NOTICE](NOTICE) — the base-model attribution, which travels with any
  redistribution of the weights.

## Licence

Apache-2.0 for the Phonon-1 family weights and this CLI; the Phonon-2 weights are CC-BY-4.0. See [LICENSE](LICENSE)
and [NOTICE](NOTICE).
