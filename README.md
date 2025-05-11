# Audio Steganography Watermark

Generative watermarks based on encryption techniques for embedding a signature in an audio file without perceptual changes being added to the content. 

## Features
- Inaudible watermarking (-64dB or lower)
- Encryption key-based watermark generation
- Null testing capability for watermark detection
- Preserves audio quality and aesthetics
- Compatible with any DAW

## Technical Specifications
- Sample Rate: 44.1kHz
- Bit Depth: 16-bit PCM
- Format: Mono WAV
- Amplitude Range: -1.0 to 1.0 (normalized)

## Why
When working with clients in the field of audio engineering it is often wise to watermark your deliverables when sharing demos. Other use cases for watermarking audio exist outside the music industry such as cybersecurity. In music you may do business with clients who you don't know very well. Reinforcing trust with these clients is paramount to ensuring that they are held accountable while you hold up your end of the deal.

## How
Watermarking an audio file presents several challenges. Many approaches are viable but each approach has different pros and cons. One could apply automations to intermittently interrupt the audio with filters or destructive processing such as waveshaping and heavy data compression using plugins such as [lossy](https://goodhertz.com/lossy/). The problem with these techniques is they alter the quality of the demo and can interrupt the listeners flow, altering their judgement of the mix. Clients may complain about the aesthetic changes, or perhaps they may actually like these changes rendering ones efforts to secure their work until payments are settled useless. 

## Solution
There exists an approach which can preserve the quality and aesthetics of the audio while still watermarking it. They key is to create an audio stem that matches the length of the track and mixing it in low enough so that it is not perceptible to the human ear (-64dB). Any audio sample may be used but having something thats random such as noise, adding modulations and automations, etc makes the watermark more unique and thus harder to recreate. You may ask how this approach is any good if the watermark is inaudible. The key is to null the original audio with the unwatermarked signal and then amplify the result to reveal whether the watermark is present or not. If not then nulling the audio content will result in absolute silence even when amplified tremendously. If it is present this indicates that the audio in question has been leaked or distributed without settling the agreement between artist and engineer. 

## Methods
In this project I present an approach using python to sonify an encryption key by converting values from the key into audio samples. This is similar to using random noise but differs in that the generated audio is the sonified encryption key. The resulting audio is a cryptographically secure random signal that can be used as a unique watermark.

## Requirements
- Python 3.8 or higher
- Audio files in WAV format
- DAW with null testing capabilities

## Installation & Generation

1. Clone this repository
```bash
git clone https://github.com/jacklion710/audio-steganography-watermark.git
cd audio-steganography-watermark
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the generation script specifying the desired duration of the watermark in mm:ss
```bash
python generate.py --duration 03:30
```

4. The resulting waveform will be rendered into the `watermarks/` directory as a 16-bit, 44.1kHz mono WAV file

5. Insert the waveform into the DAW project you wish to watermark

6. Mix the watermark so that it peaks no higher than -64dB. It can be even quieter but try to aim for -64dB. If it peaks any higher then you risk making the watermark audible which could possibly affect the aesthetics or otherwise interfere with the quality of the mix

**NOTE**: You must bypass any processing when rendering the project with the watermark. You can do this by routing all audio to a dedicated mix buss then route the watermark and mix buss to the master buss (or the dedicated main/master channel depending on your choice of DAW). This allows you to apply processing to your mix as one does when mastering while bypassing the watermark omitting it from processing. Alternatively, you can render the DAW project with post processing applied and no watermark then insert the unwatermarked version in a new project with no post processing and layer the watermark in this session. This is crucial, any processing applied to the watermark may affect your ability to isolate it when null testing.

## Watermark Detection Procedure

1. Render a version of the DAW project that does not include the watermark

2. Create a new session and insert the suspicious audio in the DAW

3. Insert the unwatermarked version that you rendered and align it with the suspicious audio

**NOTE**: Unwatermarked and suspicious audio must be exactly aligned to ensure proper null testing

4. Invert the polarity of the unwatermarked version to null test

5. Amplify the result after nulling

> ⚠️ **WARNING**: To avoid sudden and loud audio take care not to undo the null or misalign the audio samples

6. If your watermark is now audible then you can be confident that the suspicious audio was not authorized for distribution as per the agreement between the client and the engineer

7. **Optional**: If you level match the resulting null with the watermark present with the raw watermark you can null test this to ensure the same key was used.

## Troubleshooting

### Common Issues
- **Watermark is audible**: Ensure the watermark is mixed at -64dB or lower
- **Null test fails**: Check that the audio files are perfectly aligned and that no processing has been applied to the watermark
- **Generation fails**: Verify Python version and dependencies are correctly installed

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

