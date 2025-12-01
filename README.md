# Simple-English-Tutor-AI-Chat
This is a simple, open source chat setup for beginner english teacher. This was created to use as few CPU resources possible.

This repository only contains the code needed to run the model itself locally, it does not contain any additional file for front-end.
I highly recommend running it on WSL:Ubuntu and Docker.

How to set up:

project/
│── server.py
│── requirements.txt
│── Dockerfile
└── audio_test/
     └── audio_test_mono.wav
└── models/
     ├── Phi-3-mini-4k-instruct-q4.gguf
     ├── vosk-model-small-en-us-0.15/
     └── flan-t5-small/

     

1 - Install all required depencencies in a docker following the folder structure. 

2 - Build the image: 
 docker build -t tutor-server .

3 - Run the container: 
 docker run -p 8000:8000 tutor-server # Change the exposed port in the DockerFile if you want to use a different port.

4 - Send a WAV audio file to the model via Curl: 
curl -X POST \
     -F "audio=@myvoice.wav" \  # Get direct path if it doesn't find the file
     http://localhost:8000/chat-file

Thank you!
