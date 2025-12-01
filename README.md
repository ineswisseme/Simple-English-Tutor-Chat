# Simple-English-Tutor-AI-Chat



This is a simple, open source chat setup for beginner english teacher. This was created to use as few CPU resources as possible.

This repository only contains the code needed to run the model itself locally, it does not contain any additional file for front-end.
I highly recommend running it on WSL:Ubuntu and Docker.

How it works:

This chatbot uses several open source ml models. Here is a breakdown of the main loop:

1. audio.wav input sent via Curl by user.
2. pydub convert file to mono.
3. Vosk translate the audio to text.
4. T5 cleans transcription if needed.
5. Phi-3 receives user_text and produce a text answer.
6. Kokoro creates a voice over of user_text.
7. Json file containing user_text, reply_text and audio is saved in user's local directory.



How to set up:

<img width="316" height="234" alt="folder" src="https://github.com/user-attachments/assets/c410bee3-9b51-4705-a6f5-564ff2c033f0" />




1 - Create all required environment in a docker following the folder structure. The DockerFile will install the requirements.txt when you build the image.

2 - Build the image: 
 docker build -t tutor-server .

3 - Run the container: 
 docker run -p 8000:8000 tutor-server # Change the exposed port in the DockerFile if you want to use a different port.

4 - Send a WAV audio file to the model via Curl: 
curl -X POST \
     -F "audio=@myvoice.wav" \  # Get direct path if it doesn't find the file
     http://localhost:8000/chat-file

Thank you!
