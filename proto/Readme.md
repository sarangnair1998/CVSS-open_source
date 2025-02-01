# **Meeting Minutes Bot**

## **Overview**
This project is a prototype for an AI-powered Meeting Minutes Bot developed for **Creare Vincere Sea Shipping**. It automatically extracts audio from recorded meetings, transcribes speech into text using OpenAI's Whisper ASR model, summarizes the discussion using Meta-Llama-3.1-8B-Instruct, and emails the generated minutes to the recipient. While it was initially developed for **Creare Vincere Sea Shipping**, it is fully customizable and can be adapted for use by any company.

## **Features**
- Extracts audio from recorded Google Meet videos
- Transcribes speech into text using Whisper ASR
- Summarizes meeting discussions using LLaMA
- Generates structured meeting minutes in Markdown format
- Sends meeting minutes via email

## **Dependencies**
This project requires the following dependencies:

```bash
apt-get install -y ffmpeg
pip install transformers torchaudio bitsandbytes
```

## **Setup**
### **1. Mount Google Drive**
This script uses Google Drive for storing and retrieving meeting recordings. Mount Google Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### **2. Define Recording Folder**
```python
RECORDINGS_FOLDER = '/content/drive/MyDrive/Meet Recordings'
```

### **3. Extract Audio from Video**
```python
import os

entries = os.listdir(RECORDINGS_FOLDER)
video_filepath = f'/content/drive/MyDrive/Meet Recordings/{entries[6]}'
audio_filepath = '/content/drive/MyDrive/Meet Recordings/meeting_audio.mp3'

!ffmpeg -i "{video_filepath}" -q:a 0 -map a "{audio_filepath}" -y
```

## **Speech-to-Text Transcription**
Uses OpenAI's Whisper model for automatic speech recognition (ASR):

```python
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

AUDIO_MODEL = "openai/whisper-medium"
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True)
speech_model.to('cuda')
processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
```

### **Perform Transcription**
```python
asr_pipeline = pipeline("automatic-speech-recognition", model=speech_model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, torch_dtype=torch.float16, device=0, return_timestamps=True)

def transcribe_audio(audio_path):
    result = asr_pipeline(audio_path)
    return result["text"]

transcript = transcribe_audio(audio_filepath)
```

## **Meeting Minutes Generation**
Uses Meta-Llama-3.1-8B-Instruct for summarization:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")

tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL, device_map="auto", quantization_config=quant_config)
```

### **Generate Meeting Minutes**
```python
def generate_minutes(transcript_text):
    prompt = (
        "You are an assistant that produces meeting minutes from transcripts. "
        "Please generate meeting minutes in markdown format that strictly follow the template below:\n\n"
        "### Meeting Minutes\n\n"
        "**Summary:**\n"
        "- [Provide a concise summary of the meeting]\n\n"
        "**Discussion Points:**\n"
        "- [List each discussion point with details]\n\n"
        "**Takeaways:**\n"
        "- [List key takeaways from the meeting]\n\n"
        "**Action Items:**\n"
        "- [List action items along with the designated owners]\n\n"
        "Now, generate the meeting minutes for the following transcript:\n\n" + transcript_text + "\n\n### MEETING MINUTES START\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
    minutes = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return minutes

minutes_markdown = generate_minutes(transcript)
```

## **Emailing Meeting Minutes**
The generated minutes are emailed using SMTP:

```python
import smtplib
from email.mime.text import MIMEText

def send_email(subject, body, recipient, sender_email, sender_password):
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [recipient], msg.as_string())
    print(f"Email sent to {recipient}.")

SENDER_EMAIL = "your-email@gmail.com"
SENDER_PASSWORD = "your-email-password"
RECIPIENT_EMAIL = "recipient@example.com"

send_email("Meeting Minutes", minutes_markdown, RECIPIENT_EMAIL, SENDER_EMAIL, SENDER_PASSWORD)
```

## **Security Considerations**
- **DO NOT hardcode passwords in scripts**. Instead, use environment variables or a secure vault.
- **Use App Passwords** if using Gmail, as Google blocks less secure apps.
- **Avoid exposing API keys** in public repositories.

## **Future Improvements**
- Implement a UI interface for easier use.
- Improve LLaMA-based summarization quality.
- Integrate with a calendar for automatic meeting detection.

## **License**
This project is open-source and available under the MIT License.

## **Contributors**
- [Sarang Nair](https://github.com/sarangnair1998)

## **Acknowledgments**
- OpenAI Whisper for speech recognition
- Meta LLaMA for meeting summarization
- Google Colab for cloud-based execution

---

### **Developed for Creare Vincere Sea Shipping but can be adapted for any company**

