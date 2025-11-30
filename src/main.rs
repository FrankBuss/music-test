// Elevated Synth - Faithful port of the 4k intro synth by TBC and RGBA
// Original: Breakpoint 2009 winner
// Music by Puryx, Synth by Mentor

mod data;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

// Audio parameters
const SAMPLE_RATE: u32 = 44100;
const NUM_ROWS: usize = 114;
const STEPS_PER_ROW: usize = 16;
const MAX_NOTE_SAMPLES: usize = 5210;
const TOTAL_SAMPLES: usize = (NUM_ROWS * STEPS_PER_ROW * MAX_NOTE_SAMPLES + 65535) & 0xFFFF0000;
const MAX_DELAY_SAMPLES: usize = 65536;


// Machine types from machine_table in synth.asm
const MACHINE_SYNTH: u8 = 0;
const MACHINE_DELAY: u8 = 1;
const MACHINE_FILTER: u8 = 2;
const MACHINE_COMPRESSOR: u8 = 3;
const MACHINE_MIXER: u8 = 4;
const MACHINE_DISTORTION: u8 = 5;
const MACHINE_ALLPASS: u8 = 6;

// Machine data sizes (bytes)
const SIZE_SYNTH: usize = 72;      // 9 dwords ADSR/params + 3 oscillators (12 bytes each)
const SIZE_DELAY: usize = 12;      // pos_init, length, feedback
const SIZE_FILTER: usize = 32;     // cutoff, res, lfo1_freq, cos1, lfo2_freq, cos2, dry, type
const SIZE_COMPRESSOR: usize = 12; // threshold, ratio, postadd
const SIZE_MIXER: usize = 8;       // left_vol, right_vol
const SIZE_DISTORTION: usize = 8;  // a, b (sin waveshaper params)
const SIZE_ALLPASS: usize = 12;    // pos_init, length, feedback

// Envelope scales for ADSR segments
const ENVELOPE_SCALES: [f32; 4] = [1.0, -0.5, 0.0, -0.5];
const NOTESTOP_SCALES: [f32; 4] = [0.0, 0.0, 0.0, -0.5];

struct Random {
    seed: u32,
}

impl Random {
    fn new() -> Self {
        Self { seed: 1 }
    }

    fn next(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(16307).wrapping_add(17);
        ((self.seed >> 14) as i16) as f32 / 32768.0
    }
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn read_f32(data: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn osc(phase: f32, phase_shift: f32, osc_type: u8) -> f32 {
    let p = phase + phase_shift;
    let p = (p - p.round()) * 2.0;

    match osc_type {
        1 => (p * std::f32::consts::PI).sin(),
        2 => p.signum(),
        3 => p,
        _ => 0.0,
    }
}

struct FilterState {
    sin1: f32,
    sin2: f32,
    cos1: f32,
    cos2: f32,
    low: [f32; 2],
    band: [f32; 2],
}

struct DelayState {
    buf: Vec<(f32, f32)>,
    pos: usize,
}

fn generate_music() -> Vec<(f32, f32)> {
    let start_time = std::time::Instant::now();
    println!("Generating music...");

    // Frequency constants
    let c5_freq: f32 = 8.175798916;                                     // C-5 frequency in Hz
    let note_freq_start: f32 = c5_freq / SAMPLE_RATE as f32;
    let note_freq_step: f32 = 2.0_f32.powf(1.0 / 24.0);                 // Quarter tone step
    let cutoff_scale: f32 = std::f32::consts::PI / SAMPLE_RATE as f32 / 2.0;

    // Stack of audio buffers (like assembly's edi-based stack)
    let mut stack: Vec<Vec<(f32, f32)>> = vec![vec![(0.0, 0.0); TOTAL_SAMPLES]; 8];
    let mut stack_ptr: i32 = -1;

    let mut delay_bufs: Vec<DelayState> = Vec::new();
    let mut rng = Random::new();
    let mut channel = 0usize;
    let mut offset = 0usize;
    let mut machine_type = MACHINE_SYNTH;

    while offset < data::MACHINE_TREE.len() {
        if machine_type >= 128 {
            break;
        }

        match machine_type {
            MACHINE_SYNTH => {
                stack_ptr += 1;
                let level = stack_ptr as usize;
                if level >= stack.len() {
                    stack.push(vec![(0.0, 0.0); TOTAL_SAMPLES]);
                }
                for s in stack[level].iter_mut() {
                    *s = (0.0, 0.0);
                }

                let attack = read_u32(data::MACHINE_TREE, offset);
                let decay = read_u32(data::MACHINE_TREE, offset + 4);
                let sustain = read_u32(data::MACHINE_TREE, offset + 8);
                let release = read_u32(data::MACHINE_TREE, offset + 12);
                let noise_mix = read_f32(data::MACHINE_TREE, offset + 16);
                let freq_exp = read_f32(data::MACHINE_TREE, offset + 20);
                let base_freq = read_f32(data::MACHINE_TREE, offset + 24);
                let volume = read_f32(data::MACHINE_TREE, offset + 28);
                let stereo = read_f32(data::MACHINE_TREE, offset + 32);

                let mut oscs = [(0u8, 0u8, 0.0f32, 0.0f32); 3];
                for i in 0..3 {
                    let osc_off = offset + 36 + i * 12;
                    oscs[i] = (
                        data::MACHINE_TREE[osc_off],
                        data::MACHINE_TREE[osc_off + 1],
                        read_f32(data::MACHINE_TREE, osc_off + 4),
                        read_f32(data::MACHINE_TREE, osc_off + 8),
                    );
                }

                let output = &mut stack[level];
                let mut note_offset = 0usize;

                for row in 0..NUM_ROWS {
                    let seq_idx = channel * NUM_ROWS + row;
                    let pattern_idx = data::SEQUENCE_DATA.get(seq_idx).copied().unwrap_or(0) as usize;

                    for step in 0..STEPS_PER_ROW {
                        let pattern_byte = pattern_idx * STEPS_PER_ROW + step;
                        let note = data::PATTERN_DATA.get(pattern_byte).copied().unwrap_or(0);

                        let doubled_note = (note as u16) * 2;
                        let use_notestop = doubled_note == 0xFE;

                        let actual_doubled = if use_notestop {
                            if step > 0 {
                                (data::PATTERN_DATA.get(pattern_byte - 1).copied().unwrap_or(0) as u16)
                                    * 2
                            } else {
                                0
                            }
                        } else {
                            doubled_note
                        };

                        if actual_doubled > 0 {
                            let scales = if use_notestop {
                                &NOTESTOP_SCALES
                            } else {
                                &ENVELOPE_SCALES
                            };

                            let mut freq = note_freq_start;
                            for _ in 0..actual_doubled {
                                freq *= note_freq_step;
                            }
                            freq -= base_freq;

                            let mut phase = 0.0f32;
                            let mut envelope = 0.0f32;
                            let mut pos = note_offset;
                            let lengths = [
                                attack as usize,
                                decay as usize,
                                sustain as usize,
                                release as usize,
                            ];

                            for seg in 0..4 {
                                let len = lengths[seg];
                                if len == 0 {
                                    continue;
                                }
                                let env_step = scales[seg] / len as f32;

                                for _ in 0..len {
                                    if pos >= output.len() {
                                        break;
                                    }

                                    envelope += env_step;
                                    freq *= freq_exp;
                                    phase += freq + base_freq;

                                    let mut osc_sum = 0.0f32;
                                    for &(osc_type, op, phase_shift, detune) in &oscs {
                                        let o1 = osc(phase * (2.0 - detune), phase_shift, osc_type);
                                        let o2 = osc(phase * detune, phase_shift, osc_type);
                                        let val = o1 + o2;

                                        match op {
                                            1 => {}
                                            2 => osc_sum += val,
                                            3 => osc_sum -= val,
                                            4 => osc_sum *= val,
                                            _ => {}
                                        }
                                    }

                                    let noise = rng.next() * noise_mix;
                                    let sample = (osc_sum + noise) * envelope * volume;

                                    output[pos].0 = sample;
                                    output[pos].1 = sample * stereo;
                                    pos += 1;
                                }
                            }
                        }

                        note_offset += MAX_NOTE_SAMPLES;
                    }
                }

                channel += 1;
                offset += SIZE_SYNTH;
                machine_type = data::MACHINE_TREE[offset];
                offset += 1;
            }

            MACHINE_DELAY => {
                let delay_pos_init = read_u32(data::MACHINE_TREE, offset) as usize;
                let delay_len = read_u32(data::MACHINE_TREE, offset + 4) as usize;
                let feedback = read_f32(data::MACHINE_TREE, offset + 8);

                let level = stack_ptr as usize;
                let samples = &mut stack[level];

                delay_bufs.push(DelayState {
                    buf: vec![(0.0, 0.0); MAX_DELAY_SAMPLES],
                    pos: delay_pos_init,
                });
                let ds = delay_bufs.last_mut().unwrap();

                if delay_len > 0 && delay_len < ds.buf.len() {
                    for (l, r) in samples.iter_mut() {
                        ds.pos = if ds.pos == 0 {
                            delay_len - 1
                        } else {
                            ds.pos - 1
                        };
                        let (wl, wr) = ds.buf[ds.pos];
                        *l += wl * feedback;
                        *r += wr * feedback;
                        ds.buf[ds.pos] = (*l, *r);
                    }
                }

                offset += SIZE_DELAY;
                machine_type = data::MACHINE_TREE[offset];
                offset += 1;
            }

            MACHINE_FILTER => {
                let cutoff = read_f32(data::MACHINE_TREE, offset);
                let resonance = read_f32(data::MACHINE_TREE, offset + 4);
                let lfo1_freq = read_f32(data::MACHINE_TREE, offset + 8);
                let cos1_init = read_f32(data::MACHINE_TREE, offset + 12);
                let lfo2_freq = read_f32(data::MACHINE_TREE, offset + 16);
                let cos2_init = read_f32(data::MACHINE_TREE, offset + 20);
                let dry = read_f32(data::MACHINE_TREE, offset + 24);
                let filter_type = read_u32(data::MACHINE_TREE, offset + 28);

                let level = stack_ptr as usize;
                let samples = &mut stack[level];

                let mut fs = FilterState {
                    sin1: 0.0,
                    sin2: 0.0,
                    cos1: cos1_init,
                    cos2: cos2_init,
                    low: [0.0; 2],
                    band: [0.0; 2],
                };

                for (left, right) in samples.iter_mut() {
                    fs.cos1 = fs.cos1 - fs.sin1 * lfo1_freq;
                    fs.sin1 = fs.sin1 + fs.cos1 * lfo1_freq;
                    fs.cos2 = fs.cos2 - fs.sin2 * lfo2_freq;
                    fs.sin2 = fs.sin2 + fs.cos2 * lfo2_freq;

                    let f = ((fs.sin1 + fs.sin2 + cutoff) * cutoff_scale).clamp(0.0001, 0.999);

                    for (ch, sample) in
                        [(0usize, left as *mut f32), (1usize, right as *mut f32)]
                    {
                        unsafe {
                            fs.low[ch] += f * fs.band[ch];
                            let high = resonance * (*sample - fs.band[ch]) - fs.low[ch];
                            fs.band[ch] += f * high;
                            fs.band[ch] += 2.0;
                            fs.band[ch] -= 2.0;

                            let wet = match filter_type {
                                0 => fs.low[ch],
                                1 => high,
                                _ => fs.band[ch],
                            };
                            *sample = wet + dry * *sample;
                        }
                    }
                }

                offset += SIZE_FILTER;
                machine_type = data::MACHINE_TREE[offset];
                offset += 1;
            }

            MACHINE_COMPRESSOR => {
                let threshold = read_f32(data::MACHINE_TREE, offset);
                let ratio = read_f32(data::MACHINE_TREE, offset + 4);
                let postadd = read_f32(data::MACHINE_TREE, offset + 8);

                let level = stack_ptr as usize;
                let samples = &mut stack[level];

                for (l, r) in samples.iter_mut() {
                    for s in [l, r] {
                        let sign = s.signum();
                        let abs_val = s.abs();
                        let over = abs_val - threshold;
                        if over > 0.0 {
                            *s = sign * (threshold + over * ratio + postadd);
                        }
                    }
                }

                offset += SIZE_COMPRESSOR;
                machine_type = data::MACHINE_TREE[offset];
                offset += 1;
            }

            MACHINE_MIXER => {
                let left_vol = read_f32(data::MACHINE_TREE, offset);
                let right_vol = read_f32(data::MACHINE_TREE, offset + 4);

                if stack_ptr > 0 {
                    let src_level = stack_ptr as usize;
                    let dst_level = (stack_ptr - 1) as usize;

                    for i in 0..TOTAL_SAMPLES {
                        let (src_l, src_r) = stack[src_level][i];
                        let (dst_l, dst_r) = stack[dst_level][i];
                        stack[dst_level][i].0 = src_l * left_vol + dst_l * right_vol;
                        stack[dst_level][i].1 = src_r * left_vol + dst_r * right_vol;
                    }
                    stack_ptr -= 1;
                }

                offset += SIZE_MIXER;
                machine_type = data::MACHINE_TREE[offset];
                offset += 1;
            }

            MACHINE_DISTORTION => {
                let a = read_f32(data::MACHINE_TREE, offset);
                let b = read_f32(data::MACHINE_TREE, offset + 4);

                let level = stack_ptr as usize;
                for (l, r) in stack[level].iter_mut() {
                    *l = (*l * a).sin() * b;
                    *r = (*r * a).sin() * b;
                }

                offset += SIZE_DISTORTION;
                machine_type = data::MACHINE_TREE[offset];
                offset += 1;
            }

            MACHINE_ALLPASS => {
                let delay_pos_init = read_u32(data::MACHINE_TREE, offset) as usize;
                let delay_len = read_u32(data::MACHINE_TREE, offset + 4) as usize;
                let feedback = read_f32(data::MACHINE_TREE, offset + 8);

                let level = stack_ptr as usize;
                let samples = &mut stack[level];

                delay_bufs.push(DelayState {
                    buf: vec![(0.0, 0.0); MAX_DELAY_SAMPLES],
                    pos: delay_pos_init,
                });
                let ds = delay_bufs.last_mut().unwrap();

                if delay_len > 0 && delay_len < ds.buf.len() {
                    for (l, r) in samples.iter_mut() {
                        ds.pos = if ds.pos == 0 {
                            delay_len - 1
                        } else {
                            ds.pos - 1
                        };
                        let (wet_l, wet_r) = ds.buf[ds.pos];

                        let in_l = wet_r * feedback + *l;
                        let in_r = wet_l * feedback + *r;
                        ds.buf[ds.pos] = (in_l, in_r);
                        *l = wet_r - in_l * feedback;
                        *r = wet_l - in_r * feedback;
                    }
                }

                offset += SIZE_ALLPASS;
                machine_type = data::MACHINE_TREE[offset];
                offset += 1;
            }

            _ => break,
        }
    }

    // Collapse remaining stack
    while stack_ptr > 0 {
        let src = stack_ptr as usize;
        let dst = (stack_ptr - 1) as usize;
        for i in 0..TOTAL_SAMPLES {
            stack[dst][i].0 += stack[src][i].0;
            stack[dst][i].1 += stack[src][i].1;
        }
        stack_ptr -= 1;
    }

    // Normalize
    let mut max_val = 0.0f32;
    for (l, r) in stack[0].iter() {
        max_val = max_val.max(l.abs()).max(r.abs());
    }

    if max_val > 0.0 {
        let scale = 0.9 / max_val;
        for (l, r) in stack[0].iter_mut() {
            *l *= scale;
            *r *= scale;
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let duration = TOTAL_SAMPLES as f64 / SAMPLE_RATE as f64;
    println!("Generated {:.1}s of audio in {:.2}s ({:.1}x realtime)", duration, elapsed, duration / elapsed);
    stack.remove(0)
}

fn main() {
    println!("Elevated Synth - Rust Port");
    println!("Original: TBC & RGBA, Breakpoint 2009\n");

    let audio_data = Arc::new(generate_music());
    let playback_pos = Arc::new(Mutex::new(0usize));

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No output device available");

    println!("\nUsing audio device: {}", device.name().unwrap_or_default());

    let config = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let audio_data_clone = Arc::clone(&audio_data);
    let playback_pos_clone = Arc::clone(&playback_pos);

    let stream = device
        .build_output_stream(
            &config,
            move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut pos = playback_pos_clone.lock().unwrap();
                for frame in output.chunks_mut(2) {
                    if *pos < audio_data_clone.len() {
                        let (l, r) = audio_data_clone[*pos];
                        frame[0] = l;
                        frame[1] = r;
                        *pos += 1;
                    } else {
                        frame[0] = 0.0;
                        frame[1] = 0.0;
                    }
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )
        .expect("Failed to build output stream");

    stream.play().expect("Failed to start playback");

    println!("\nPlaying... Press Ctrl+C to stop.\n");

    let duration_secs = audio_data.len() as f64 / SAMPLE_RATE as f64;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let pos = *playback_pos.lock().unwrap();
        let current_secs = pos as f64 / SAMPLE_RATE as f64;
        print!(
            "\r{:02}:{:05.2} / {:02}:{:05.2}  ",
            (current_secs / 60.0) as u32,
            current_secs % 60.0,
            (duration_secs / 60.0) as u32,
            duration_secs % 60.0
        );
        use std::io::Write;
        std::io::stdout().flush().ok();

        if pos >= audio_data.len() {
            println!("\nPlayback complete.");
            break;
        }
    }
}
