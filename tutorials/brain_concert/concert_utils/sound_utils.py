"""Toy piano-synthesizer utilities used by the "Brain Concert" bonus demo.

This module generates raw audio waveforms for musical notes using simple
signal-processing techniques (additive synthesis, ADSR-style envelopes,
and Karplus-Strong string synthesis), and returns them as
``pydub.AudioSegment`` objects that can be concatenated/overlaid to build
a melody.

Typical usage::

    song = AudioSegment.silent(duration=0)
    song_left = AudioSegment.silent(duration=0)
    note_freq = get_scale("C", "major", 4)
    note_freq_left = get_scale("C", "major", 3)

    for note, beats in melody:
        duration = beats / BPS
        song += generate_piano_note(note_freq[note], duration)
        song_left += generate_piano_note(note_freq_left[note], duration)

    full_song = song.overlay(song_left - 6)
    song.export("output.wav", format="wav")
"""

import numpy as np
from pydub import AudioSegment

SAMPLE_RATE = 44100  # Standard CD-quality sample rate (Hz)

# Frequency (Hz) of every note/octave combination used by this module,
# following standard equal-temperament tuning (A4 = 440 Hz).
ALL_NOTES = {
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81,
    "F3": 174.61, "F#3": 185.00, "G3": 196.00, "G#3": 207.65, "A3": 220.00,
    "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63,
    "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00,
    "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "D5": 587.33, "E5": 659.25, "F5": 698.46, "G5": 783.99,
    "A5": 880.00, "B5": 987.77,
}

# Semitone offsets from the tonic that define each supported scale.
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],  # tone-tone-semitone-tone-tone-tone-semitone
    "minor": [0, 2, 3, 5, 7, 8, 10],  # tone-semitone-tone-tone-semitone-tone-tone
    "pentatonic": [0, 2, 4, 7, 9],  # major pentatonic example
}


def get_scale(root: str, scale_type: str = "major", octave: int = 4) -> dict[str, float]:
    """Build a note-name -> frequency mapping for one musical scale.

    Args:
        root: Tonic note name without octave (e.g. ``"C"``), case-insensitive.
        scale_type: Scale type, one of the keys of :data:`SCALE_INTERVALS`
            (``"major"``, ``"minor"``, or ``"pentatonic"``).
        octave: Octave of the tonic note. Notes that cross into the next
            octave (e.g. the 7th degree of a scale rooted near the top of
            an octave) are automatically resolved to ``octave + 1``.

    Returns:
        Dictionary mapping each scale-degree note name (without octave,
        e.g. ``"C"``, ``"D"``, ...) to its frequency in Hz.

    Raises:
        ValueError: If ``root`` is not a valid note name.
        KeyError: If ``scale_type`` is not a supported scale.
    """
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    root_name = root.upper()
    root_index = note_names.index(root_name)

    intervals = SCALE_INTERVALS[scale_type]
    scale_notes: dict[str, float] = {}
    for interval in intervals:
        note_index = (root_index + interval) % 12
        note_octave = octave + (root_index + interval) // 12
        note_name = note_names[note_index]
        full_note_name = f"{note_name}{note_octave}"
        scale_notes[note_name] = ALL_NOTES[full_note_name]
    return scale_notes


def generate_simple_note(freq: float, duration: float) -> AudioSegment:
    """Synthesize a pure sine-wave tone.

    Args:
        freq: Fundamental frequency of the note, in Hz.
        duration: Note duration, in seconds.

    Returns:
        A mono 16-bit ``AudioSegment`` containing the synthesized tone.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(freq * t * 2 * np.pi)

    audio = wave * (2**15 - 1)
    audio = audio.astype(np.int16)

    return AudioSegment(
        audio.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1,
    )


def generate_piano_note(freq: float, duration: float) -> AudioSegment:
    """Synthesize a simple piano-like tone via additive synthesis.

    Combines the fundamental frequency with three decaying harmonic
    overtones, then applies a fast-attack/slow-decay amplitude envelope
    to approximate the percussive character of a struck piano string.

    Args:
        freq: Fundamental frequency of the note, in Hz.
        duration: Note duration, in seconds.

    Returns:
        A mono 16-bit ``AudioSegment`` containing the synthesized tone.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Additive synthesis: fundamental + decaying harmonics (piano-like timbre).
    wave = (
        1.0 * np.sin(2 * np.pi * freq * t)
        + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.25 * np.sin(2 * np.pi * 3 * freq * t)
        + 0.1 * np.sin(2 * np.pi * 4 * freq * t)
    )

    # Optional: small detune to simulate multiple strings
    # detune = np.random.uniform(-0.003, 0.003)
    # wave += 0.3 * np.sin(2 * np.pi * freq * (1 + detune) * t)

    # Piano envelope: near-instant attack, exponential decay.
    attack = int(0.01 * SAMPLE_RATE)
    decay = np.exp(-3 * t)

    envelope = decay
    envelope[:attack] = np.linspace(0, 1, attack)

    wave = wave * envelope

    # Normalize to full scale before quantizing to int16.
    wave = wave / np.max(np.abs(wave))

    audio = (wave * (2**15 - 1)).astype(np.int16)

    return AudioSegment(
        audio.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1,
    )


def generate_piano_note_ym(freq: float, duration: float) -> AudioSegment:
    """Synthesize a richer piano-like tone with hammer noise and echo/reverb.

    Extends :func:`generate_piano_note` with:
        - Slightly detuned (inharmonic) overtones, simulating imperfect
          string coupling.
        - Two additional slightly-detuned unison strings.
        - A short burst of filtered noise to approximate the hammer strike.
        - A randomized attack/decay envelope for a more natural, less
          mechanical sound.
        - A simple multi-tap echo/reverb effect.

    Args:
        freq: Fundamental frequency of the note, in Hz.
        duration: Note duration, in seconds.

    Returns:
        A mono 16-bit ``AudioSegment`` containing the synthesized tone.

    Side Effects:
        Uses NumPy's global RNG (``np.random``), so repeated calls with the
        same arguments will *not* produce identical output.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Main harmonics with small inharmonicity (detuned overtone ratios),
    # mimicking the slightly stretched harmonic series of a real piano string.
    wave = (
        1.0 * np.sin(2 * np.pi * freq * t)
        + 0.6 * np.sin(2 * np.pi * 2.01 * freq * t)
        + 0.4 * np.sin(2 * np.pi * 3.02 * freq * t)
        + 0.25 * np.sin(2 * np.pi * 4.05 * freq * t)
        + 0.1 * np.sin(2 * np.pi * 5.1 * freq * t)
    )

    # Slightly detuned unison strings (simulates multiple physical strings
    # per key, as in a real piano).
    wave += 0.25 * np.sin(2 * np.pi * freq * 1.002 * t)
    wave += 0.25 * np.sin(2 * np.pi * freq * 0.998 * t)

    # Short burst of filtered noise at note onset, approximating the
    # hammer-strike transient.
    noise = np.random.normal(0, np.random.uniform(0.015, 0.025), len(t))
    wave += noise * np.exp(-40 * t)

    # Randomized attack/decay envelope: fast attack, blended
    # fast-decay + slow-decay tail (adds a natural "ringing" quality).
    attack = int(np.random.uniform(0.001, 0.003) * SAMPLE_RATE)
    decay_high = np.exp(-np.random.uniform(4, 6) * t)
    decay_low = np.exp(-0.5 * t)  # longer decay tail, feeds the echo below
    envelope = 0.6 * decay_high + 0.4 * decay_low
    envelope[:attack] = np.linspace(0, 1, attack)
    wave *= envelope

    # Multi-tap echo/reverb: sum of delayed, attenuated copies of the signal.
    reverb = np.zeros_like(wave)
    delays = [0.02, 0.04, 0.07, 0.11]  # seconds
    gains = [0.2, 0.15, 0.1, 0.05]  # per-tap attenuation
    for delay_seconds, gain in zip(delays, gains):
        delay_samples = int(delay_seconds * SAMPLE_RATE)
        reverb[delay_samples:] += gain * wave[:-delay_samples]

    # Additional cross-string "sympathetic" echo (~30ms).
    reverb += 0.05 * np.roll(wave, int(0.03 * SAMPLE_RATE))

    wave += reverb

    # Normalize to full scale before quantizing to int16.
    wave /= np.max(np.abs(wave))

    audio = (wave * (2**15 - 1)).astype(np.int16)
    return AudioSegment(
        audio.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1,
    )


def generate_arpa_note(freq: float, duration: float) -> AudioSegment:
    """Synthesize a plucked-string tone using Karplus-Strong synthesis.

    Initializes a short ring buffer with white noise, then repeatedly
    averages adjacent samples (a simple low-pass filter) while feeding the
    buffer back into itself, producing a decaying, harmonically-rich tone
    reminiscent of a plucked string or harp note.

    Args:
        freq: Fundamental frequency of the note, in Hz. Determines the
            ring-buffer length (``SAMPLE_RATE / freq`` samples).
        duration: Note duration, in seconds.

    Returns:
        A mono 16-bit ``AudioSegment`` containing the synthesized tone.

    Side Effects:
        Uses NumPy's global RNG (``np.random``) to seed the initial noise
        burst, so repeated calls with the same arguments will *not*
        produce identical output.
    """
    buffer_size = int(SAMPLE_RATE / freq)
    # Initialize the "string" with white noise.
    ring_buffer = np.random.uniform(-1, 1, buffer_size)

    num_samples = int(SAMPLE_RATE * duration)
    output = np.zeros(num_samples)

    for i in range(num_samples):
        output[i] = ring_buffer[i % buffer_size]
        # Karplus-Strong update: average two adjacent samples and apply a
        # slight damping factor, which low-pass filters and decays the
        # signal over time.
        ring_buffer[i % buffer_size] = 0.5 * (ring_buffer[i % buffer_size] + ring_buffer[(i + 1) % buffer_size]) * 0.996

    # Normalize to full scale before quantizing to int16.
    output /= np.max(np.abs(output))

    audio = (output * (2**15 - 1)).astype(np.int16)

    return AudioSegment(
        audio.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1,
    )