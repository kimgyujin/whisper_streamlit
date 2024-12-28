#import
from typing import Dict
from uuid import uuid4
from io import BytesIO
import os
import tempfile

import numpy as np
import whisper
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, WebSocket
from pydub import AudioSegment
from pydantic import BaseModel
from pytube import YouTube

app = FastAPI()

# schema for youtube transcription api
class YoutubeUrl(BaseModel):
    youtube_url: str
    
tasks = {}

# load whisper model
model = whisper.load_model("turbo")

def create_task_id() -> str:
    return str(uuid4())

@app.get("/status/{task_id}")
async def check_status(task_id: str) -> Dict[str, str]:
    task = tasks.get(task_id)
    
    if task:
        return task
    
    return {"status": "not found"}

async def process_local_audio_file(task_id: str, upload_file: UploadFile) -> None:
    try:
        # extract audio byte from file
        audio_bytes = await upload_file.read()
        
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        raw_data = audio.raw_data
        
        audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        audio_np /= np.iinfo(np.int16).max # normalization
        
        # whisper 실행
        result = model.transcribe(audio_np)
        transcription = result.get("text")
        
        # task status update
        tasks[task_id] = {"status": "completed", "transcription": transcription}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}

def download_youtube_audio(youtube_url: str) -> str:
    try:
        yt = YouTube(youtube_url)
        
        audio_stream = (
            yt.streams.filter(only_audio=True).order_by("bitrate").desc().first()
        )
        
        # 다운로드된 파일 저장하기 위해 임시 저장        
        temp_dir = tempfile.gettempdir()
        temp_filename = audio_stream.default_filename
        temp_filepath = os.path.join(temp_dir, temp_filename)
        
        # 임시 파일로부터 audio stream 다운로드
        audio_stream.download(output_path=temp_dir, filename=temp_filename)
        
        return temp_filepath
    except Exception as e:
        raise Exception(f"Error downloading YouTube audio: {e}")

def process_youtube_audio(task_id: str, file_path: str) -> None:
    try:
        result = model.transcribe(file_path)
        transcription = result.get("text")
        
        tasks[task_id] = {"status": "completed", "transcription": transcription}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/transcribe-local/")
async def transcribe_local_file(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> Dict[str, str]:
    # task id 생성
    task_id = create_task_id()
    
    # task status 초기화
    tasks[task_id] = {"status": "processing"}
    
    # audio processing, transcribe 추가
    background_tasks.add_task(process_local_audio_file, task_id, file)
    
    # status 추적을 위한 task id 반환
    return {"task_id": task_id}

@app.post("/transcribe-youtube/")
async def transcribe_youtbe(
    background_tasks: BackgroundTasks, youtube_url: YoutubeUrl
) -> Dict[str, str]:
    task_id = create_task_id()
    
    tasks[task_id] = {"status": "processing"}
    
    background_tasks.add_task(process_youtube_audio, task_id, download_youtube_audio(youtube_url.youtube_url))
    
    return {"task_id": task_id}

@app.websocket("/ws")
async def transcribe_websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            
            updated_data = np.frombuffer(data, dtype=np.float32).copy()
            transcription = model.transcribe(updated_data)
            
            await websocket.send_text(transcription["text"])
    except Exception as e:
        print(f"Error in WebSocket communication: {e}")
    finally:
        await websocket.close()