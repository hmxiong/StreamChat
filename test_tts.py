import time
import torch
from TTS.api import TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
time_0 = time.time()
# tts = TTS(model_path="/13390024681/All_Model_Zoo/Tacotron2-DDC", config_path="/13390024681/All_Model_Zoo/Tacotron2-DDC/config.json").to(device)
# tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(device)
tts = TTS(model_path="/13390024681/All_Model_Zoo/XTTS-v2",config_path="/13390024681/All_Model_Zoo/XTTS-v2/config.json", progress_bar=False).to(device)
time_1 = time.time()

tts.tts_to_file(text="Could you please tell me what you see from these images?", 
                speaker_wav="/13390024681/llama/EfficientVideo/Ours/test_1.wav", 
                language="en",
                file_path="output.wav")

time_2 = time.time()
print("spend time:{}, {}".format((time_1-time_0), (time_2 - time_1)))
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts

# config = XttsConfig()
# config.load_json("/13390024681/All_Model_Zoo/XTTS-v2/config.json")
# model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="/13390024681/All_Model_Zoo/XTTS-v2", eval=True)
# model.cuda()

# outputs = model.synthesize(
#     "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
#     config,
#     gpt_cond_len=3,
#     language="en",
# )

# # Import necessary libraries and configure settings
# import torch
# import torchaudio
# torch._dynamo.config.cache_size_limit = 64
# torch._dynamo.config.suppress_errors = True
# torch.set_float32_matmul_precision('high')

# import ChatTTS
# from IPython.display import Audio

# # Initialize and load the model: 
# chat = ChatTTS.Chat()
# chat.load_models(source='custom',
#                  custom_path='/13390024681/All_Model_Zoo/ChatTTS',
#                  compile=False) # Set to True for better performance

# # Define the text input for inference (Support Batching)
# texts = [
#     "So we found being competitive and collaborative was a huge way of staying motivated towards our goals, so one person to call when you fall off, one person who gets you back on then one person to actually do the activity with.",
#     ]

# # Perform inference and play the generated audio
# wavs = chat.infer(texts)
# Audio(wavs[0], rate=24_000, autoplay=True)

# # Save the generated audio 
# torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)