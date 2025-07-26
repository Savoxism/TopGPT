from googleapiclient.discovery import build
from tqdm import tqdm
import pandas as pd
import isodate
from dotenv import load_dotenv
import os

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def search(query, max_results=50):
    vids, token = [], None
    pbar = tqdm(total=max_results, desc=query)
    while len(vids) < max_results:
        resp = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoDuration="long",
            maxResults=min(50, max_results - len(vids)),
            pageToken=token
        ).execute()
        for it in resp["items"]:
            vid_id = it["id"]["videoId"]
            vids.append({
                "videoId": vid_id,
                "title": it["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "publishedAt": it["snippet"]["publishedAt"]
            })
            pbar.update(1)
        token = resp.get("nextPageToken")
        if not token:
            break
    pbar.close()
    return vids

queries = [
    "Andrew Tate podcast",
    "Andrew Tate interview",
    "Andrew Tate full interview",
    "Andrew Tate long-form video",
    "Andrew Tate discussion",
    "Andrew Tate guest appearance",
    "Andrew Tate Emergency Meetings",
    "Andrew Tate Emergency Meetings FULL",
    "Andrew Tate interview 2024",
    "Andrew Tate interview 2023",
    "Andrew Tate interview 2025",
]


all_v = []
for q in queries:
    all_v.extend(search(q, max_results=10))

df = (
    pd.DataFrame(all_v)
      .drop_duplicates("videoId")
      .reset_index(drop=True)
)

df['publishedAt'] = pd.to_datetime(df['publishedAt'])
df_sorted = df.sort_values('publishedAt', ascending=False).reset_index(drop=True)

def filter_min_duration(df, min_seconds=1800):
    keep_ids = []
    for i in tqdm(range(0, len(df), 50), desc="Filtering by duration"):
        batch_ids = df["videoId"].iloc[i:i+50].tolist()
        resp = youtube.videos().list(
            part="contentDetails",
            id=",".join(batch_ids)
        ).execute()
        for item in resp.get("items", []):
            content = item.get("contentDetails", {})
            duration = content.get("duration")
            if duration:
                dur_s = isodate.parse_duration(duration).total_seconds()
                if dur_s >= min_seconds:
                    keep_ids.append(item["id"])
    return df[df["videoId"].isin(keep_ids)].copy()

df_filtered = filter_min_duration(df_sorted)

df_filtered.to_csv("andrew_tate_long_form_raw", index=False)
print(f"Saved {len(df_filtered)} videos ≥ 30 min to CSV.")