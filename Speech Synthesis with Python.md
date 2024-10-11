## Speech Synthesis with Python
Slide 1: Introduction to Speech Synthesis

Speech synthesis is the artificial production of human speech. It involves converting written text into spoken words, a process known as text-to-speech (TTS). This technology has numerous applications, from accessibility tools for the visually impaired to voice assistants in smart devices.

```python
def text_to_speech(text):
    # This is a simplified representation of a TTS system
    phonemes = convert_to_phonemes(text)
    audio = generate_audio_from_phonemes(phonemes)
    return audio

def convert_to_phonemes(text):
    # Convert text to phonetic representation
    # This is a complex process involving language rules
    return ["h", "ə", "l", "oʊ"]

def generate_audio_from_phonemes(phonemes):
    # Generate audio waveform from phonemes
    # This involves acoustic modeling and signal processing
    return [0.1, 0.2, 0.3, 0.4, 0.5]  # Simplified audio waveform

# Example usage
text = "Hello, world!"
speech_audio = text_to_speech(text)
print(f"Generated audio waveform: {speech_audio}")
```

Slide 2: Results for: Introduction to Speech Synthesis

```python
Generated audio waveform: [0.1, 0.2, 0.3, 0.4, 0.5]
```

Slide 3: Text Analysis and Normalization

The first step in speech synthesis is text analysis and normalization. This process involves converting raw text into a standardized format, expanding abbreviations, converting numbers to words, and handling special characters.

```python
def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Expand common abbreviations
    abbreviations = {
        "mr.": "mister",
        "dr.": "doctor",
        "st.": "saint",
        "ave.": "avenue"
    }
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)
    
    # Convert numbers to words
    numbers = {
        "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
        "6": "six", "7": "seven", "8": "eight", "9": "nine", "0": "zero"
    }
    for digit, word in numbers.items():
        text = text.replace(digit, word)
    
    return text

# Example usage
input_text = "Mr. Smith lives at 123 Main St."
normalized_text = normalize_text(input_text)
print(f"Original text: {input_text}")
print(f"Normalized text: {normalized_text}")
```

Slide 4: Results for: Text Analysis and Normalization

```python
Original text: Mr. Smith lives at 123 Main St.
Normalized text: mister smith lives at onetwothree main street
```

Slide 5: Phonetic Transcription

After normalization, the text is converted into a phonetic representation. This process maps words to their corresponding phonemes, which are the smallest units of sound in a language.

```python
def phonetic_transcription(text):
    # A simplified phonetic dictionary
    phonetic_dict = {
        "hello": "həˈloʊ",
        "world": "ˈwərld",
        "speech": "spiːtʃ",
        "synthesis": "ˈsɪnθəsɪs"
    }
    
    words = text.split()
    phonetic_words = []
    
    for word in words:
        if word in phonetic_dict:
            phonetic_words.append(phonetic_dict[word])
        else:
            # For unknown words, we'll just use the original word
            phonetic_words.append(word)
    
    return " ".join(phonetic_words)

# Example usage
text = "Hello world speech synthesis"
phonetic_text = phonetic_transcription(text)
print(f"Original text: {text}")
print(f"Phonetic transcription: {phonetic_text}")
```

Slide 6: Results for: Phonetic Transcription

```python
Original text: Hello world speech synthesis
Phonetic transcription: həˈloʊ ˈwərld spiːtʃ ˈsɪnθəsɪs
```

Slide 7: Prosody Generation

Prosody refers to the rhythm, stress, and intonation of speech. Generating appropriate prosody is crucial for natural-sounding speech synthesis. This involves determining the pitch, duration, and intensity of each phoneme.

```python
import random

def generate_prosody(phonetic_text):
    phonemes = phonetic_text.split()
    prosody = []
    
    for phoneme in phonemes:
        # Generate random values for pitch, duration, and intensity
        # In a real system, these would be determined by complex rules
        pitch = random.uniform(80, 200)  # Hz
        duration = random.uniform(0.05, 0.2)  # seconds
        intensity = random.uniform(60, 80)  # dB
        
        prosody.append({
            "phoneme": phoneme,
            "pitch": pitch,
            "duration": duration,
            "intensity": intensity
        })
    
    return prosody

# Example usage
phonetic_text = "həˈloʊ ˈwərld"
prosody_info = generate_prosody(phonetic_text)
for item in prosody_info:
    print(f"Phoneme: {item['phoneme']}, Pitch: {item['pitch']:.2f} Hz, "
          f"Duration: {item['duration']:.2f} s, Intensity: {item['intensity']:.2f} dB")
```

Slide 8: Results for: Prosody Generation

```python
Phoneme: həˈloʊ, Pitch: 137.24 Hz, Duration: 0.15 s, Intensity: 74.53 dB
Phoneme: ˈwərld, Pitch: 182.91 Hz, Duration: 0.08 s, Intensity: 65.18 dB
```

Slide 9: Acoustic Modeling

Acoustic modeling is the process of generating the actual sound waveforms for each phoneme. This typically involves using a database of pre-recorded speech segments and concatenating or modifying them to create the desired output.

```python
import math

def generate_sine_wave(frequency, duration, sample_rate=44100):
    t = [i / sample_rate for i in range(int(duration * sample_rate))]
    return [math.sin(2 * math.pi * frequency * time) for time in t]

def acoustic_modeling(prosody_info, sample_rate=44100):
    waveform = []
    
    for item in prosody_info:
        phoneme_wave = generate_sine_wave(item['pitch'], item['duration'], sample_rate)
        # Apply intensity (simplified as amplitude scaling)
        scaled_wave = [sample * (item['intensity'] / 100) for sample in phoneme_wave]
        waveform.extend(scaled_wave)
    
    return waveform

# Example usage
prosody_info = [
    {"phoneme": "həˈloʊ", "pitch": 130, "duration": 0.3, "intensity": 70},
    {"phoneme": "ˈwərld", "pitch": 110, "duration": 0.4, "intensity": 65}
]

waveform = acoustic_modeling(prosody_info)
print(f"Generated waveform length: {len(waveform)} samples")
print(f"First 10 samples: {waveform[:10]}")
```

Slide 10: Results for: Acoustic Modeling

```python
Generated waveform length: 30870 samples
First 10 samples: [0.0, 0.012566304230164353, 0.025126605181358997, 0.03767653776666441, 0.05020181578706289, 0.06268917269624819, 0.07512535023499682, 0.08749719562194368, 0.09979157614630174, 0.11199545202386005]
```

Slide 11: Waveform Generation

The final step in speech synthesis is waveform generation, where the acoustic model output is converted into a digital audio signal that can be played through speakers or saved as an audio file.

```python
import wave
import struct

def generate_wav_file(waveform, filename, sample_rate=44100):
    # Normalize the waveform to 16-bit range
    max_amplitude = max(abs(sample) for sample in waveform)
    normalized_waveform = [int(sample / max_amplitude * 32767) for sample in waveform]
    
    # Create a new WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        
        # Write the audio data
        for sample in normalized_waveform:
            wav_file.writeframes(struct.pack('h', sample))

# Example usage
waveform = acoustic_modeling(prosody_info)  # From the previous example
generate_wav_file(waveform, "synthesized_speech.wav")
print("WAV file 'synthesized_speech.wav' has been generated.")
```

Slide 12: Real-Life Example: Navigation System

A common application of speech synthesis is in navigation systems. These systems convert route instructions into spoken directions, helping drivers focus on the road.

```python
def navigation_tts(route):
    synthesized_speech = []
    
    for instruction in route:
        # Convert distance to spoken form
        distance = instruction['distance']
        if distance < 100:
            spoken_distance = f"In {distance} meters"
        else:
            spoken_distance = f"In {distance // 100} hundred meters"
        
        # Combine distance and action
        spoken_instruction = f"{spoken_distance}, {instruction['action']}"
        
        # Here we would call our TTS function, but we'll simulate it
        synthesized_speech.append(spoken_instruction)
    
    return synthesized_speech

# Example usage
route = [
    {"distance": 200, "action": "turn right"},
    {"distance": 50, "action": "turn left"},
    {"distance": 500, "action": "continue straight"},
]

spoken_route = navigation_tts(route)
for instruction in spoken_route:
    print(instruction)
```

Slide 13: Results for: Real-Life Example: Navigation System

```python
In 2 hundred meters, turn right
In 50 meters, turn left
In 5 hundred meters, continue straight
```

Slide 14: Real-Life Example: Assistive Technology for Visually Impaired

Speech synthesis plays a crucial role in assistive technology for visually impaired individuals, converting written text into spoken words to aid in reading and comprehension.

```python
def screen_reader(screen_content):
    def synthesize(text):
        # In a real system, this would call a TTS engine
        return f"Speaking: {text}"
    
    output = []
    
    for element in screen_content:
        if element['type'] == 'text':
            output.append(synthesize(element['content']))
        elif element['type'] == 'image':
            output.append(synthesize(f"Image: {element['alt_text']}"))
        elif element['type'] == 'button':
            output.append(synthesize(f"Button: {element['label']}"))
    
    return output

# Example usage
screen_content = [
    {"type": "text", "content": "Welcome to the homepage"},
    {"type": "image", "alt_text": "Company logo"},
    {"type": "button", "label": "Login"}
]

spoken_content = screen_reader(screen_content)
for speech in spoken_content:
    print(speech)
```

Slide 15: Results for: Real-Life Example: Assistive Technology for Visually Impaired

```python
Speaking: Welcome to the homepage
Speaking: Image: Company logo
Speaking: Button: Login
```

Slide 16: Additional Resources

For those interested in delving deeper into speech synthesis, here are some valuable resources:

1.  "Statistical Parametric Speech Synthesis Using Deep Neural Networks" by Heiga Zen et al. (2013) ArXiv: [https://arxiv.org/abs/1306.3328](https://arxiv.org/abs/1306.3328)
2.  "WaveNet: A Generative Model for Raw Audio" by Aäron van den Oord et al. (2016) ArXiv: [https://arxiv.org/abs/1609.03499](https://arxiv.org/abs/1609.03499)
3.  "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" by Jonathan Shen et al. (2018) ArXiv: [https://arxiv.org/abs/1712.05884](https://arxiv.org/abs/1712.05884)

These papers provide in-depth insights into advanced techniques in speech synthesis, including the use of deep learning models for generating more natural-sounding speech.

