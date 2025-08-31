import os
import uuid
import shutil
import asyncio
from collections import deque
from tempfile import gettempdir
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from huggingface_hub import HfApi
from fastapi.middleware.cors import CORSMiddleware
import json
from huggingface_hub import HfApi, CommitOperationAdd

# --------------------
# CONFIG
# --------------------

AUDIO_DIR = os.path.join(gettempdir(), "audio_chunks")  # cross‑platform temp dir
DATASET_DIR = os.path.join(gettempdir(), "dataset")
URLS_FILE = "urls.txt"
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO = "1219Naveen/Asr_data"
KV_FILE = os.path.join(DATASET_DIR, "kv.jsonl")  # local working copy we keep in sync
kv_lock = asyncio.Lock()  # to prevent concurrent writes to kv.jsonl

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

queue = deque()
queue_lock = asyncio.Lock()
api = HfApi(token=HF_TOKEN)


class TranscriptSubmission(BaseModel):
    clip_id: str
    transcript: str


# --------------------
# UTILITIES
# --------------------
async def extract_audio_chunks(url: str):
    """Download ONE video/audio, convert to 16k mono WAV, split into 30s chunks, enqueue them.
    This function does not look at other URLs and is only called on demand.
    """
    import yt_dlp
    import subprocess

    # 1) Download best audio with proper extension
    ydl_outtmpl = os.path.join(AUDIO_DIR, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": ydl_outtmpl,
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        src_audio = ydl.prepare_filename(info)  # resolves to actual downloaded file path

    # 2) Work inside a unique batch dir for this URL
    batch_dir = os.path.join(AUDIO_DIR, f"batch_{info.get('id', uuid.uuid4().hex)}")
    os.makedirs(batch_dir, exist_ok=True)

    # 3) Convert to clean 16kHz mono WAV (robust for ASR)
    wav_file = os.path.join(batch_dir, "audio.wav")
    cmd_convert = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", src_audio,
        "-vn", "-ac", "1", "-ar", "16000", "-y",
        wav_file,
    ]
    subprocess.run(cmd_convert, check=True)

    # Optionally remove the source file to save space
    try:
        os.remove(src_audio)
    except OSError:
        pass

    # 4) Segment into 30s chunks
    chunk_pattern = os.path.join(batch_dir, "chunk_%03d.wav")
    cmd_segment = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", wav_file,
        "-f", "segment",
        "-segment_time", "30",
        "-reset_timestamps", "1",
        chunk_pattern,
    ]
    subprocess.run(cmd_segment, check=True)

    # Remove the big intermediate wav to save disk
    try:
        os.remove(wav_file)
    except OSError:
        pass

    # 5) Enqueue each chunk with a unique clip_id embedded in the filename for easy requeue
    async with queue_lock:
        for fname in sorted(os.listdir(batch_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            old_path = os.path.join(batch_dir, fname)
            clip_id = str(uuid.uuid4())
            new_path = os.path.join(batch_dir, f"{clip_id}.wav")
            os.rename(old_path, new_path)
            queue.append({
                "clip_id": clip_id,
                "path": new_path,
                "batch_dir": batch_dir,
                "url": url,
                "status": "pending",
            })
from fastapi.responses import FileResponse
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/clip/{clip_id}")
async def get_clip(clip_id: str):
    """Return the actual audio file for playback."""
    for root, _, files in os.walk(AUDIO_DIR):
        for f in files:
            if f.startswith(clip_id) and f.lower().endswith(".wav"):
                return FileResponse(os.path.join(root, f), media_type="audio/wav")
    return JSONResponse(status_code=404, content={"status": "error", "msg": "clip not found"})


def get_next_youtube_link():
    """Pop exactly ONE URL (first line) from urls.txt and rewrite the file without it."""
    if not os.path.exists(URLS_FILE):
        return None

    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f.readlines() if u.strip()]

    if not urls:
        return None

    next_url = urls[0]
    with open(URLS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(urls[1:]))
    return next_url


# --------------------
# FASTAPI APP
# --------------------

@app.get("/next-clip")
async def next_clip():
    async with queue_lock:
        if not queue:
            video_url = get_next_youtube_link()
        else:
            video_url = None

    if video_url:
        try:
            await extract_audio_chunks(video_url)
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "status": "failed",
                "reason": str(e),
            })

    async with queue_lock:
        if not queue:
            return {"status": "no_more_links_or_no_chunks"}
        item = queue.popleft()
        item["status"] = "in_progress"
        return {
            "clip_id": item["clip_id"],
            "download_url": f"http://localhost:8000/clip/{item['clip_id']}"
        }

@app.post("/submit")
async def submit(submission: TranscriptSubmission):
    """
    Accept transcript for a clip, upload to HF as a key-value mapping:
      - key:  audio/<clip_id>.wav  (the audio file path inside the repo)
      - value: transcript string   (stored as a line in kv.jsonl)
    Also cleans up local files after a successful commit.
    """
    # 1) Locate the clip by filename
    clip_path = None
    for root, _, files in os.walk(AUDIO_DIR):
        for f in files:
            if f.startswith(submission.clip_id) and f.lower().endswith(".wav"):
                clip_path = os.path.join(root, f)
                break
        if clip_path:
            break

    if not clip_path or not os.path.exists(clip_path):
        return JSONResponse(status_code=404, content={"status": "error", "msg": "clip not found"})

    # 2) Decide final repo path for this audio file
    repo_audio_path = f"audio/{submission.clip_id}.wav"

    # 3) Stage a local copy of the audio to upload (kept outside the served queue dir)
    #    Use a tiny staging dir to avoid name collisions and to clean easily.
    stage_dir = os.path.join(DATASET_DIR, f"stage_{submission.clip_id}")
    os.makedirs(stage_dir, exist_ok=True)
    staged_audio_path = os.path.join(stage_dir, f"{submission.clip_id}.wav")
    shutil.copy(clip_path, staged_audio_path)

    # 4) Append/maintain the local kv.jsonl (append-only)
    #    Each line: {"key": "audio/<clip_id>.wav", "value": "<transcript>"}
    async with kv_lock:
        os.makedirs(DATASET_DIR, exist_ok=True)
        with open(KV_FILE, "a", encoding="utf-8") as kvf:
            kvf.write(json.dumps({"key": repo_audio_path, "value": submission.transcript}, ensure_ascii=False))
            kvf.write("\n")

    # 5) Commit both files to HF in a single commit
    operations = [
        CommitOperationAdd(path_in_repo=repo_audio_path, path_or_fileobj=staged_audio_path),
        CommitOperationAdd(path_in_repo="kv.jsonl", path_or_fileobj=KV_FILE),
    ]

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: api.create_commit(
                repo_id=HF_REPO,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Add {repo_audio_path} and update kv.jsonl",
            )
        )
    except Exception as e:
        # Do not delete the clip if commit failed; keep it so the user can retry/requeue
        # Also revert last appended line in local KV_FILE to avoid drift if needed
        # (simple, safe approach: leave the line; next commit will upload the current KV_FILE again)
        shutil.rmtree(stage_dir, ignore_errors=True)
        return JSONResponse(status_code=500, content={"status": "failed", "reason": str(e)})

    # 6) Cleanup local staging + served chunk; optionally delete empty batch dir
    shutil.rmtree(stage_dir, ignore_errors=True)

    try:
        os.remove(clip_path)
    except OSError:
        pass

    batch_dir = os.path.dirname(clip_path)
    try:
        if os.path.isdir(batch_dir) and not any(os.scandir(batch_dir)):
            shutil.rmtree(batch_dir, ignore_errors=True)
    except OSError:
        pass

    return {
        "status": "uploaded",
        "clip_id": submission.clip_id,
        "key": repo_audio_path,
        "value_preview": submission.transcript[:80] + ("…" if len(submission.transcript) > 80 else "")
    }



@app.post("/requeue/{clip_id}")
async def requeue(clip_id: str):
    """If the client couldn't transcribe the served clip, push it back to the queue."""
    # Find the file again by name (clip_id.wav)
    clip_path = None
    for root, _, files in os.walk(AUDIO_DIR):
        for f in files:
            if f.startswith(clip_id) and f.lower().endswith(".wav"):
                clip_path = os.path.join(root, f)
                break
        if clip_path:
            break

    if not clip_path or not os.path.exists(clip_path):
        return JSONResponse(status_code=404, content={"status": "error", "msg": "clip not found"})

    async with queue_lock:
        queue.append({
            "clip_id": clip_id,
            "path": clip_path,
            "batch_dir": os.path.dirname(clip_path),
            "url": "",
            "status": "pending",
        })
    return {"status": "requeued", "clip_id": clip_id}


# (Optional) small debug endpoints
@app.get("/queue-size")
async def queue_size():
    async with queue_lock:
        return {"size": len(queue)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)