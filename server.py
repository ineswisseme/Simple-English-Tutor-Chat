import os
import io
import json
import time
import logging
import wave as wave_module
import base64
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from llama_cpp import Llama
from vosk import Model, KaldiRecognizer
from kokoro import KPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI() # keeping fastAPI in case user wants to make their own http server if wanted. 


WORKSPACE = Path("/workspace")        
WORKSPACE.mkdir(exist_ok=True)        



logger.info("Loading Flan-T5-small")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
flan_model.eval()

logger.info("Loading Phi-3-mini")
phi3_model = Llama(
    model_path="/models/Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=20,
    n_ctx=4096,
    n_threads=4
)

logger.info("Loading Vosk STT.")
vosk_model = Model("/models/vosk-model-small-en-us-0.15")

logger.info("Loading Kokoro TTS")
pipeline = KPipeline(lang_code='a')

logger.info("All models loaded successfully.")



def clean_transcription_with_flan(text: str) -> str:
    prompt = (
        "Fix this speech-to-text transcription. "
        "Correct errors but keep meaning the same. "
        "Return only corrected text:\n"
        f"{text}"
    )
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = flan_model.generate(
            **inputs, max_new_tokens=30, temperature=0.1, top_p=0.9, num_beams=4
        )
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def clean_sentence_artifacts(text: str) -> str:
    return text.replace("=== Sentence 1:", "").replace("=== Sentence 2:", "").strip()

def enforce_two_sentences_and_question(text: str) -> str:
    text = clean_sentence_artifacts(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    while len(sentences) < 3:
        sentences.append("")

    s1 = sentences[0].rstrip(".!?") + "."
    s2 = sentences[1].rstrip(".!?") + "."
    q = sentences[2].rstrip(".!?") + "?"

    return f"{s1} {s2} {q}"

def build_chat_prompt(user_text: str) -> str:
    return (
        "<|system|>\n"
        "You are an English tutor helping Patrick learn English.\n"
        "Rules: exactly TWO short sentences, then ONE short question.\n"
        "<|user|>\n"
        f"{user_text}\n"
        "<|assistant|>\n"
    )

def run_phi3(user_text: str) -> str:
    prompt = build_chat_prompt(user_text)
    response = phi3_model(
        prompt,
        max_tokens=40,
        temperature=0.1,
        top_p=0.1,
        repeat_penalty=1.1,
        stop=["<|user|>", "<|assistant|>", "<|system|>"]
    )
    return response["choices"][0]["text"].strip()



@app.post("/chat-file")
async def chat_file(audio: UploadFile = File(...)):
    logger.info("Received /chat-file request")

    # Read audio
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        buffer = io.BytesIO(content)
        audio_seg = AudioSegment.from_file(buffer)
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    samplerate = audio_seg.frame_rate
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)

    if audio_seg.channels > 1:
        samples = samples.reshape((-1, audio_seg.channels)).mean(axis=1)

    samples /= np.abs(samples).max() or 1.0

    wav_buf = io.BytesIO()
    sf.write(wav_buf, samples, samplerate, format="WAV")
    wav_buf.seek(0)

    # Vosk STT:
    wf = wave_module.open(wav_buf, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    audio_data = wf.readframes(wf.getnframes())
    result = json.loads(rec.AcceptWaveform(audio_data) and rec.Result() or rec.FinalResult())
    raw_text = result.get("text", "")

    logger.info(f"Transcribed: '{raw_text}'")

    # Clean text with T5
    user_text = clean_transcription_with_flan(raw_text)

    # Phi-3 reply
    try:
        reply_raw = run_phi3(user_text)
        reply = enforce_two_sentences_and_question(reply_raw)
        logger.info(f"LLM reply: '{reply}'")
    except Exception:
        raise HTTPException(status_code=500, detail="LLM generation failed")

    # TTS from kokoro:
    try:
        generator = pipeline(reply, voice="af_sarah", speed=1.0)
        chunks = [a for _, _, a in generator]
        sr = 24000

        audio_out = np.concatenate(chunks, axis=0)
        buf = io.BytesIO()
        sf.write(buf, audio_out, sr, format='WAV')
        buf.seek(0)

        audio_bytes = buf.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    except Exception:
        raise HTTPException(status_code=500, detail="TTS failed.")



    json_path = WORKSPACE / f"generated_text.json"
    audio_path = WORKSPACE / f"generated_audio.wav"

    # Save JSON check your workspace folder
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "user_text": user_text,
            "reply_text": reply,
            "audio_base64": audio_b64
        }, f, ensure_ascii=False, indent=2)

    # Save audio WAV same path
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)


    return {"status": "audio and text saved"}
