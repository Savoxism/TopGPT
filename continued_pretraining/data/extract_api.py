import os
import pandas as pd
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

output_dir = "tate_long_form_data"
os.makedirs(output_dir, exist_ok=True)

def safe_filename(title):
    safe = re.sub(r'[\\/*?:"<>|]', "", title)
    safe = safe.strip()[:180]  # limit length to avoid OS errors
    return safe

file_path = "data/andrew_tate/andrew_tate_long_form_raw.csv"
df = pd.read_csv(file_path)

ytt_api = YouTubeTranscriptApi()
skipped = 0
success = 0
errors = 0

for idx, row in df.iterrows():
    url = row['url']
    video_id = url.split("v=")[-1].split("&")[0]
    title = row['title']
    fname = safe_filename(title) + ".txt"
    out_path = os.path.join(output_dir, fname)

    # Skip already extracted
    if os.path.exists(out_path):
        print(f"[{idx}] Already exists: {fname}, skipping.")
        skipped += 1
        continue

    print(f"\nProcessing row {idx}:")
    print(f"URL: {url}")
    print(f"Extracted video_id: {video_id}")

    try:
        print(f"Fetching transcript for video_id: {video_id} ...")
        fetched_transcript = ytt_api.fetch(video_id)
        print(f"Fetched transcript snippet count: {len(fetched_transcript)}")
        full_text = " ".join(snippet.text for snippet in fetched_transcript)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Saved: {fname}")
        success += 1
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(f"Transcript not available for {video_id}: {title} — {e}")
        errors += 1
    except Exception as e:
        print(f"Error for {video_id}: {title} — {e}")
        errors += 1

print(f"\nExtraction finished: {success} transcripts saved, {skipped} skipped, {errors} errors.")