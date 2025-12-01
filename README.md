# Simple-English-Tutor-AI-Chat



This is a simple, open source chat setup for beginner english teacher. This was created to use as few CPU resources as possible.

This repository only contains the code needed to run the model itself locally, it does not contain any additional file for front-end.
I highly recommend running it on WSL:Ubuntu. You will need Docker.

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

<img width="319" height="255" alt="folder" src="https://github.com/user-attachments/assets/d11d4eb2-c412-4252-a08d-5d2a780275a6" />



1 - Create all required environment in a docker following the folder structure. The DockerFile will install the requirements.txt when you build the image, this includes the workspace and models folders.

2 - Build the image: 

 docker build -t tutor-server .

3 - Run the container: 

 docker run --name tutor   -p port:port   -v ~/workspace:/workspace   tutor-server 
 # Choose the exposed port in the DockerFile. 

4 - Send a WAV audio file to the model via Curl: 

curl -X POST   -F "audio=@ ~/audio_test/output_mono_argentina.wav"   http://localhost:*your port*/chat-file

Thank you!
