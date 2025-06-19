from flask import Flask, request, jsonify, send_from_directory
import os
import yt_dlp
import whisper
from moviepy.editor import VideoFileClip

app = Flask(__name__)
UPLOAD_DIR = "clips"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = whisper.load_model("base")

def download_video(url):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(UPLOAD_DIR, '%(id)s.%(ext)s'),
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return os.path.join(UPLOAD_DIR, f"{info['id']}.mp4"), info['id']

@app.route("/api/process", methods=["POST"])
def process_video():
    data = request.get_json()
    video_url = data.get("url")

    if not video_url:
        return jsonify({"error": "Missing video URL"}), 400

    try:
        video_path, vid_id = download_video(video_url)
        result = model.transcribe(video_path)
        text = result['text']

        clip_start = result['segments'][0]['start']
        clip_end = result['segments'][0]['end']
        
        clip_file = f"{vid_id}_clip.mp4"
        clip_path = os.path.join(UPLOAD_DIR, clip_file)

        clip = VideoFileClip(video_path).subclip(clip_start, clip_end)
        clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")

        return jsonify({
            "text": text,
            "clips": [{
                "clip_url": f"/clips/{clip_file}",
                "start": clip_start,
                "end": clip_end
            }]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clips/<filename>")
def get_clip(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
