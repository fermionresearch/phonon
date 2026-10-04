# Hotwords

Hotwords are names and terms you want the model to get right: people, products, places, project names, jargon.
Give Phonon-2 up to 25 of them and the decoder favours them whenever the audio is close. The audio still decides:
a listed word that was not said stays out of the transcript.

Hotwords work on every Phonon-2 engine: Apple silicon (MLX), the CPU engine on Linux, Windows and macOS, and the
NVIDIA CUDA image. Without hotwords the transcript is exactly the one you get today.

## Command line

```bash
fermion transcribe phonon-2 call.wav --hotwords "Ada, Quillon, Neutrino"
fermion transcribe phonon-2 call.wav --hotwords "Ada Lovelace" --hotwords Quillon  # repeatable
fermion transcribe phonon-2 call.wav --hotwords @names.txt                          # or --hotwords-file names.txt
fermion listen phonon-2 --hotwords "Ada, Quillon"
```

A hotwords file holds one term per line (or comma-separated terms); `#` starts a comment. Multi-word terms such
as `Ada Lovelace` are kept whole.

`--hotword-lambda` sets how hard the listed words pull (default 2.0). Raise it a little when a listed word is still
missed; lower it if listed words start to appear where something else was said.

## Python

```python
import fermion

speech = fermion.load_speech("phonon-2")
text, decode_s, audio_s = speech.transcribe("call.wav", hotwords=["Ada", "Quillon", "Neutrino"])

result = speech.transcribe_detailed("call.wav", hotwords=["Ada"], hotword_lambda=2.5)
print(result.text, result.words)
```

`transcribe`, `transcribe_detailed` and `transcribe_array` (16 kHz mono float32) take `hotwords=[...]` and
`hotword_lambda=` for that call. `speech.set_hotwords([...], lam=2.0)` sets them for every later call, and
`speech.set_hotwords([])` turns them off. A term can also be given with the ways people say it, which are favoured
too: `{"word": "Quillon", "spoken": ["quillon", "kwil on"]}`.

## Server

`fermion serve phonon-2` (and the CUDA image's `serve`) take hotwords per request on
`/v1/audio/transcriptions`:

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@call.wav" -F "model=phonon-2" \
  -F "hotwords=Ada, Quillon, Neutrino"
```

OpenAI clients can send the same list as `prompt`, which Phonon reads as a vocabulary list:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
with open("call.wav", "rb") as f:
    out = client.audio.transcriptions.create(model="phonon-2", file=f, prompt="Ada, Quillon, Neutrino")
print(out.text)
```

`prompt` is split on commas when it contains one (`Ada Lovelace, Quillon`), otherwise on spaces
(`Ada Quillon Neutrino`). An explicit `hotwords` field (comma-separated, a JSON list, or one `hotwords[]` part per
term) wins over `prompt`; `hotword_lambda` sets the strength. Each request's hotwords apply to that request only.

## How the words are favoured

Each listed word is spelled out in the model's word pieces, in every way the vocabulary can spell it, as written,
in lower case and capitalised. While decoding, a piece that continues one of those spellings gets a fixed bonus
(`hotword_lambda`) on the decoder's score for that step. Nothing is ever penalised, the "no word here" choice is
never boosted, and the timing of each word is read from the unbiased scores, so word timestamps keep their meaning.

## Good to know

- At most 25 terms per decode. Extra terms are dropped, in the order given, with a note.
- Short, common-sounding words are the hardest to pull in; longer and more distinctive terms respond best.
- Phonon-1 models take `--hotwords` on Apple silicon with their own strength scale, `--hotword-strength`.
