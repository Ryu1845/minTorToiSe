# minTorToiSe
A minimal PyTorch re-implementation of TorToiSe-TTS inference

# How to run
Only supports python between 3.8 and 3.10 included
```bash
git clone https://github.com/Ryu1845/minTorToiSe/
python -m venv .venv
source .venv/bin/activate
pip install -e .
python tortoise/inference.py --text "Tortoise is a text-to-speech program that is capable of synthesizing speech in multiple voices with realistic prosody and intonation." --conditioning_speech "emma.wav" --n_timestep 80
```
