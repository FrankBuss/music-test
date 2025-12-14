# Rust music test

Rust port of the synthesizer from **Elevated**, the legendary 4k intro by TBC and RGBA that won Breakpoint 2009.

- **Music by:** Puryx
- **Original synth by:** Mentor

## Links

- [Elevated 4k Demo (YouTube)](https://www.youtube.com/watch?v=jB0vBmiTr6o) - Watch the full original demo
- [Original Source Code (GitHub)](https://github.com/in4k/rgba_tbc_elevated_source) - The original assembly source

## Building and running

```bash
cargo run --release
```

## Synth Architecture

The synthesizer uses a **stack-based machine tree** where instruments and effects are chained together. Each machine processes audio and passes it through the stack.

### Machines

#### Synth (Sound Generator)
The core instrument that generates tones from oscillators.

| Parameter | Type | Description |
|-----------|------|-------------|
| Attack | u32 | Attack time in samples |
| Decay | u32 | Decay time in samples |
| Sustain | u32 | Sustain time in samples |
| Release | u32 | Release time in samples |
| Noise Mix | f32 | Amount of white noise mixed with oscillators |
| Freq Exp | f32 | Frequency exponential modifier (pitch slide) |
| Base Freq | f32 | Base frequency offset |
| Volume | f32 | Output volume |
| Stereo | f32 | Stereo spread (right channel multiplier) |

Each synth has **3 oscillators** with these parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| Type | u8 | Waveform: 1=sine, 2=square, 3=sawtooth |
| Operation | u8 | Combine mode: 1=none, 2=add, 3=subtract, 4=multiply |
| Phase Shift | f32 | Phase offset (0.0 to 1.0) |
| Detune | f32 | Detuning amount |

Each oscillator produces **two voices** detuned against each other (`phase × (2-detune)` and `phase × detune`), creating a richer sound. The oscillators are combined using their operation modes.

#### Filter (State-Variable Filter)
A resonant filter with dual LFO modulation.

| Parameter | Type | Description |
|-----------|------|-------------|
| Cutoff | f32 | Base cutoff frequency |
| Resonance | f32 | Resonance amount |
| LFO1 Freq | f32 | First LFO frequency |
| LFO1 Phase | f32 | First LFO initial phase (cosine) |
| LFO2 Freq | f32 | Second LFO frequency |
| LFO2 Phase | f32 | Second LFO initial phase (cosine) |
| Dry | f32 | Dry/wet mix (0=fully wet) |
| Type | u32 | Filter type: 0=lowpass, 1=highpass, 2=bandpass |

The two LFOs modulate the cutoff frequency, allowing for evolving timbres.

#### Delay
Echo effect with feedback.

| Parameter | Type | Description |
|-----------|------|-------------|
| Position | u32 | Initial buffer position |
| Length | u32 | Delay time in samples |
| Feedback | f32 | Feedback amount (0.0 to 1.0) |

#### Allpass
Allpass filter used for reverb-like diffusion. Cross-feeds left and right channels for stereo width.

| Parameter | Type | Description |
|-----------|------|-------------|
| Position | u32 | Initial buffer position |
| Length | u32 | Delay length in samples |
| Feedback | f32 | Feedback coefficient |

#### Distortion
Sine waveshaper for harmonic distortion.

| Parameter | Type | Description |
|-----------|------|-------------|
| A | f32 | Drive amount (scales input to sin function) |
| B | f32 | Output gain |

Formula: `output = sin(input × A) × B`

#### Compressor
Dynamics processor to control volume peaks.

| Parameter | Type | Description |
|-----------|------|-------------|
| Threshold | f32 | Compression threshold |
| Ratio | f32 | Compression ratio |
| Post-add | f32 | Makeup gain added after compression |

#### Mixer
Combines audio from the stack.

| Parameter | Type | Description |
|-----------|------|-------------|
| Left Vol | f32 | Left channel volume |
| Right Vol | f32 | Right channel volume |

Mixes the current stack level with the previous level using the specified volumes.

### Signal Flow

1. **Synth** machines generate audio and push it onto the stack
2. **Effects** (Filter, Delay, Allpass, Distortion, Compressor) process the top of the stack in-place
3. **Mixer** combines stack levels, reducing the stack
4. Final output is normalized to prevent clipping
