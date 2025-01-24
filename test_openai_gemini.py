from openai import OpenAI
import base64
import cv2
from moviepy.editor import VideoFileClip
import time
import os
import google.generativeai as genai
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# We'll be using the OpenAI DevDay Keynote Recap video. You can review the video here: https://www.youtube.com/watch?v=h02ti0Bl6zk
VIDEO_PATH = "data/keynote_recap.mp4"
IMAGE_PATH = "image_path"

# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)

## Set the API key
client = OpenAI(api_key="your_api_key_here")

MODEL="gpt-4o"

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path

# Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)

def openai_response_for_videos():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
        {"role": "system", "content": "You are generating a video summary. Please provide a summary of the video. Respond in Markdown."},
        {"role": "user", "content": [
            "These are the frames from the video.",
            *map(lambda x: {"type": "image_url", 
                            "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
            ],
        }
        ],
        temperature=0,
    )
    print(response.choices[0].message.content)

def openai_response_for_images():
    completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that helps me with my math homework!"},
        {"role": "user", "content": "Hello! Could you solve 20 x 5?"}
    ]
    )
    print("Assistant: " + completion.choices[0].message.content)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
            {"role": "user", "content": [
                {"type": "text", "text": "What's the area of the shape in this image?"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    print(response.choices[0].message.content)

def gemini_response_for_video():
    video_file_name = "BigBuckBunny_320x180.mp4"

    print(f"Uploading file...")
    video_file = genai.upload_file(path=video_file_name) # upload file for later analyze
    print(f"Completed upload: {video_file.uri}") # 50 minet limit for video uploading
    
    while video_file.state.name == "PROCESSING":
        print('Waiting for video to be processed.')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    print(f'Video processing complete: ' + video_file.uri)
    
    # Create the prompt.
    prompt = "Describe this video."

    # Set the model to Gemini 1.5 Flash.
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    # Make the LLM request.
    print("Making LLM inference request...")
    response = model.generate_content([prompt, video_file],
                                    request_options={"timeout": 600})
    print(response.text)
    
def vertext_response_for_video():
    import vertexai

    from vertexai.generative_models import GenerativeModel, Part

    # TODO(developer): Update and un-comment below line
    # project_id = "PROJECT_ID"

    vertexai.init(project=project_id, location="us-central1")

    vision_model = GenerativeModel(model_name="gemini-1.0-pro-vision-001")

    # Generate text
    response = vision_model.generate_content(
        [
            Part.from_uri(
                "gs://cloud-samples-data/video/animals.mp4", mime_type="video/mp4"
            ),
            "What is in the video?",
        ]
    )
    print(response.text)
