#!/usr/bin/env python3

import time
from gtts import gTTS
import os
import vlc
import threading
from functools import wraps

def run_in_thread(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
    return wrapper

@run_in_thread
def text2Speech(text, lang='en'):
    # Create a gTTS object
    tts = gTTS(text=text, lang=lang)
    
    # Save the audio file
    tts.save("output.mp3")

    p = vlc.MediaPlayer("output.mp3")    
    p.play()
    
    # Making sure the audio finishes
    time.sleep(1.5)
    duration = p.get_length() / 1000
    time.sleep(duration)
    os.remove("output.mp3")
    

# Example usage
def main():
    text = "Hello, world! This is a test of text-to-speech conversion in Python."
    text2Speech(text)

    for i in range(10):
        print(i)




if __name__ == "__main__":
    main()
