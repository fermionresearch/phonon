# Model cards

Phonon ships as three models, and each one carries its own model card on
Hugging Face. The cards are maintained there, so this file points at them
rather than keeping a copy that could fall out of date.

| Model | Download | On disk | Full LibriSpeech clean / other | Macro WER (8 benchmarks) | Card |
|---|---:|---:|:---:|---:|---|
| **Phonon-1** — the flagship and the CLI default. Across five real-world benchmarks — AMI (meetings), Earnings-22 (earnings calls), GigaSpeech (web video), SPGISpeech (financial speech), TED-LIUM (talks) — no downloadable model we could find is both smaller (download bytes) and more accurate, on any of the five. | 415.1 MB | 0.455 GB | 2.640 % / 5.699 % | 7.671 | <https://huggingface.co/FermionResearch/Phonon-1> |
| **Phonon-1 Big** — the largest build. | 580.9 MB | 0.822 GB | 2.667 % / 5.722 % | 7.604 | <https://huggingface.co/FermionResearch/Phonon-1-Big> |
| **Phonon-1 Micro** — the smallest install; beats Moonshine base on all eight benchmarks. | 285.1 MB | 0.331 GB | 3.002 % / 6.511 % | 8.522 | <https://huggingface.co/FermionResearch/Phonon-1-Micro> |

LibriSpeech figures are the full test set (5,559 utterances); the macro spans
eight public benchmarks — one protocol, standard normalized WER. Each card
carries that model's full evaluation, licence and attribution.

## In this repo instead

- [README](README.md#benchmarks) — the three models side by side on the full
  benchmark table.
- [README](README.md#run-it) — install and the exact commands.
- [NOTICE](NOTICE) — the base-model attribution, which travels with any
  redistribution of the weights.

## Licence

Apache-2.0 for the weights and this CLI. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). The base model is Apache-2.0.
