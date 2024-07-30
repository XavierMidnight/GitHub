@echo off
conda create -n soundmaker python
conda activate soundmaker
pip install -r requirements.txt
pip install torch torchaudio einops stable_audio_tools
pause