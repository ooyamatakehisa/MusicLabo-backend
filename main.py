from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import librosa
import soundfile
import numpy as np

app = FastAPI()

root = "/api/v1"


@app.get(f"{root}/")
async def root():
    return {"message": "Hello World"}


@app.post(f"{root}/pitch_shift/")
async def create_shifted_audio(n_steps: int, file: UploadFile = File(...)):
    y, sr = librosa.core.load(file, sr=None, mono=False)
    y_shift0 = librosa.effects.pitch_shift(y[0], sr, n_steps)
    y_shift1 = librosa.effects.pitch_shift(y[1], sr, n_steps)
    y_shift = np.array([y_shift0, y_shift1])
    path = "~/new.wav"
    soundfile.write(path, y_shift, sr)
    return FileResponse(path)


@app.post(f"{root}/time_stretch/")
async def create_stretched_audio(rate: int, file: UploadFile = File(...)):
    y, sr = librosa.core.load(file, sr=None, mono=False)
    y_stretched0 = librosa.effects.pitch_shift(y[0], rate)
    y_stretched1 = librosa.effects.pitch_shift(y[1], rate)
    y_shift = np.array([y_stretched0, y_stretched1])
    path = "~/new.wav"
    soundfile.write(path, y_shift, sr)
    return FileResponse(path)
