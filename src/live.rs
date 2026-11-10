//! Transcription during the meeting, on the remote worker: every few seconds
//! the speech that has finished since the last pass goes out, so when the
//! recording stops only its last stretch is left to transcribe.
//!
//! The regions are found the same way as after the meeting, over the audio so
//! far. Later audio can move them a little (the noise floor settles), so what
//! is done live is a head start, not the answer: after the meeting, a region
//! the live passes covered keeps their words and any other region goes
//! through whisper then (see `transcribe::side_pass`).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gtk::glib;

use crate::APP_NAME;
use crate::audio::{CHANNELS, RATE};
use crate::remote::Remote;
use crate::transcribe::{self, FRAME, Glued, Region, TAPS, WHISPER_RATE, Word};

/// What one side's live passes transcribed, on the real timeline.
#[derive(Debug, Default, Clone)]
pub struct SidePrior {
    /// The regions the passes held, as (start, end) in samples.
    pub covered: Vec<(usize, usize)>,
    /// Their words, with times in ms of the recording.
    pub words: Vec<Word>,
}

#[derive(Debug, Default, Clone)]
pub struct Prior {
    pub mic: SidePrior,
    pub computer: SidePrior,
    /// What whisper found when the language was left on "auto".
    pub language: Option<String>,
}

impl Prior {
    fn seconds(&self) -> f64 {
        let side = |s: &SidePrior| s.covered.iter().map(|(a, b)| b - a).sum::<usize>();
        (side(&self.mic) + side(&self.computer)) as f64 / WHISPER_RATE as f64
    }
}

/// How often the recording is looked at.
const STEP: Duration = Duration::from_secs(15);
/// A region is finished once this much quiet follows it.
const SETTLED: usize = WHISPER_RATE * 2;
/// A pass goes out once this much finished speech waits...
const BATCH: usize = WHISPER_RATE * 20;
/// ...or once the oldest of it has waited this long.
const WAIT: usize = WHISPER_RATE * 45;
/// Audio kept before the last region sent: a region found later may start a
/// little earlier than it did before.
const KEEP: usize = WHISPER_RATE * 60;
/// After a failed pass, wait this long before the next try.
const BACK_OFF: Duration = Duration::from_secs(30);

/// One side's audio so far, 16 kHz mono.
#[derive(Default)]
struct Track {
    /// Which sample `samples[0]` is.
    base: usize,
    samples: Vec<f32>,
    /// The level of every whole 30 ms frame since the start.
    levels: Vec<f32>,
    peak: f32,
}

impl Track {
    fn push(&mut self, new: &[f32]) {
        self.samples.extend_from_slice(new);
        self.peak = new.iter().fold(self.peak, |m, s| m.max(s.abs()));
        while (self.levels.len() + 1) * FRAME <= self.base + self.samples.len() {
            let from = self.levels.len() * FRAME - self.base;
            self.levels
                .push(transcribe::rms(&self.samples[from..from + FRAME]));
        }
    }

    /// Forgets the audio before `sample`, except what the levels still need.
    fn trim(&mut self, sample: usize) {
        let keep_from = sample.min(self.levels.len() * FRAME);
        if keep_from > self.base {
            self.samples.drain(..keep_from - self.base);
            self.base = keep_from;
        }
    }

    /// The sound of `region` as the mix after the meeting has it.
    fn sound(&self, region: &Region, gain: f32) -> Vec<f32> {
        self.samples[region.start - self.base..region.end - self.base]
            .iter()
            .map(|s| transcribe::soft_clip(s * gain))
            .collect()
    }
}

#[derive(Default)]
struct Side {
    track: Track,
    /// Where the last region sent ends.
    done: usize,
    prior: SidePrior,
    /// Segment numbers given out so far, so every pass gets its own.
    segments: usize,
}

/// The live passes over a recording that keeps growing.
pub struct Live {
    remote: Option<Remote>,
    language: String,
    mic: Side,
    computer: Side,
}

impl Live {
    pub fn new(remote: Option<Remote>, language: &str) -> Self {
        Live {
            remote,
            language: language.to_owned(),
            mic: Side::default(),
            computer: Side::default(),
        }
    }

    /// Adds the audio recorded since the last call, 16 kHz mono.
    pub fn push(&mut self, mic: &[f32], computer: &[f32]) {
        self.mic.track.push(mic);
        self.computer.track.push(computer);
    }

    /// Sends what has finished and waited long enough. Returns the seconds of
    /// audio transcribed.
    pub fn step(&mut self) -> Result<f64, String> {
        let frames = self
            .mic
            .track
            .levels
            .len()
            .min(self.computer.track.levels.len());
        if frames == 0 {
            return Ok(0.0);
        }
        let len = frames * FRAME;
        // The regions of the mix after the meeting: each track brought to its
        // speaking level first, here by the level of the audio so far.
        let gains = (
            transcribe::level_gain(&self.mic.track.levels, self.mic.track.peak),
            transcribe::level_gain(&self.computer.track.levels, self.computer.track.peak),
        );
        let scaled = |levels: &[f32], gain: f32| -> Vec<f32> {
            levels[..frames].iter().map(|l| l * gain).collect()
        };
        let mic_levels = scaled(&self.mic.track.levels, gains.0);
        let computer_levels = scaled(&self.computer.track.levels, gains.1);
        let mic_regions = transcribe::own_speech_regions_of(&mic_levels, &computer_levels, len);
        let computer_regions = transcribe::speech_regions_of(&[&computer_levels], len);

        if self.remote.is_none() {
            self.remote = Remote::connect()?;
        }
        let Some(remote) = self.remote.clone() else {
            return Err("no worker is set".into());
        };
        let mut seconds = 0.0;
        for (side, regions, gain) in [
            (&mut self.mic, mic_regions, gains.0),
            (&mut self.computer, computer_regions, gains.1),
        ] {
            seconds += pass(&remote, &mut self.language, side, regions, gain, len)?;
        }
        Ok(seconds)
    }

    pub fn prior(&self) -> Prior {
        Prior {
            mic: self.mic.prior.clone(),
            computer: self.computer.prior.clone(),
            language: (self.language != "auto").then(|| self.language.clone()),
        }
    }
}

/// One side's pass over its finished regions, when there are enough of them.
fn pass(
    remote: &Remote,
    language: &mut String,
    side: &mut Side,
    regions: Vec<Region>,
    gain: f32,
    len: usize,
) -> Result<f64, String> {
    let from = side.done.max(side.track.base);
    let ready: Vec<Region> = regions
        .into_iter()
        .filter(|r| r.start >= from && r.end + SETTLED <= len)
        .collect();
    let Some(first) = ready.first() else {
        return Ok(0.0);
    };
    let waiting: usize = ready.iter().map(|r| r.end - r.start).sum();
    if waiting < BATCH && len - first.end < WAIT {
        return Ok(0.0);
    }
    let glued = Glued::glue(&ready, |r| side.track.sound(r, gain));
    let (words, found) = remote.words(&glued.samples, language)?;
    if *language == "auto"
        && let Some(found) = found
    {
        *language = found;
    }
    let base = side.segments;
    side.segments += words.iter().map(|w| w.segment + 1).max().unwrap_or(0);
    side.prior.words.extend(words.into_iter().map(|w| Word {
        start_ms: glued.locate(w.start_ms).0,
        end_ms: glued.locate(w.end_ms).0,
        segment: base + w.segment,
        ..w
    }));
    side.prior
        .covered
        .extend(ready.iter().map(|r| (r.start, r.end)));
    side.done = ready.last().map_or(side.done, |r| r.end);
    side.track.trim(side.done.saturating_sub(KEEP));
    Ok(waiting as f64 / WHISPER_RATE as f64)
}

// ---------------------------------------------------------------------------
// Following the recording on disk

/// Reads a staging `.raw` file (s16le, 48 kHz, stereo) as it grows, as
/// 16 kHz mono.
struct Follower {
    path: PathBuf,
    offset: u64,
    /// Bytes of an unfinished frame.
    partial: Vec<u8>,
    resampler: Resampler,
}

impl Follower {
    fn new(path: PathBuf) -> Self {
        Follower {
            path,
            offset: 0,
            partial: Vec::new(),
            resampler: Resampler::new(RATE as usize / WHISPER_RATE),
        }
    }

    fn read(&mut self) -> Vec<f32> {
        let Ok(mut file) = File::open(&self.path) else {
            return Vec::new();
        };
        let mut bytes = std::mem::take(&mut self.partial);
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return Vec::new();
        }
        let before = bytes.len();
        if file.read_to_end(&mut bytes).is_err() {
            bytes.truncate(before);
        }
        self.offset += (bytes.len() - before) as u64;
        let frame = 2 * CHANNELS as usize;
        let whole = bytes.len() / frame * frame;
        self.partial = bytes[whole..].to_vec();
        let mono: Vec<f32> = bytes[..whole]
            .chunks_exact(frame)
            .map(|f| {
                let sum: f32 = f
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|s| f32::from(i16::from_le_bytes(*s)))
                    .sum();
                sum / CHANNELS as f32 / 32768.0
            })
            .collect();
        self.resampler.feed(&mono)
    }
}

/// `transcribe::downsample` for audio that keeps coming: the same filter,
/// the same samples out.
struct Resampler {
    factor: usize,
    kernel: Vec<f32>,
    /// Input from sample `start` on.
    input: Vec<f32>,
    start: usize,
    /// The next sample out.
    next: usize,
}

impl Resampler {
    fn new(factor: usize) -> Self {
        Resampler {
            factor,
            kernel: transcribe::downsample_kernel(factor),
            input: Vec::new(),
            start: 0,
            next: 0,
        }
    }

    fn feed(&mut self, new: &[f32]) -> Vec<f32> {
        self.input.extend_from_slice(new);
        let half = TAPS / 2;
        let end = self.start + self.input.len();
        let mut out = Vec::new();
        while self.next * self.factor + half < end {
            let center = self.next * self.factor;
            let mut sum = 0.0;
            for (k, weight) in self.kernel.iter().enumerate() {
                let i = center as isize + k as isize - half as isize;
                if i >= 0 {
                    sum += self.input[i as usize - self.start] * weight;
                }
            }
            out.push(sum);
            self.next += 1;
        }
        let keep_from = (self.next * self.factor).saturating_sub(half);
        if keep_from > self.start {
            self.input.drain(..keep_from - self.start);
            self.start = keep_from;
        }
        out
    }
}

/// The live passes over a recording in progress, on a thread of their own.
pub struct Handle {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Prior>>,
}

/// Starts following the recording in `staging`, when a worker is set.
pub fn spawn(staging: &Path, language: &str) -> Option<Handle> {
    crate::remote::configured()?;
    let stop = Arc::new(AtomicBool::new(false));
    let (flag, staging, language) = (stop.clone(), staging.to_owned(), language.to_owned());
    let thread = std::thread::Builder::new()
        .name("live-transcription".into())
        .spawn(move || follow(&staging, &language, &flag))
        .ok()?;
    Some(Handle {
        stop,
        thread: Some(thread),
    })
}

impl Handle {
    /// Stops following and returns what was transcribed. A pass under way is
    /// finished first.
    pub fn finish(mut self) -> Prior {
        self.stop.store(true, Ordering::Relaxed);
        self.thread
            .take()
            .and_then(|thread| thread.join().ok())
            .unwrap_or_default()
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn follow(staging: &Path, language: &str, stop: &AtomicBool) -> Prior {
    let mut live = Live::new(None, language);
    let mut mic = Follower::new(staging.join("mic.raw"));
    let mut computer = Follower::new(staging.join("system.raw"));
    let mut next = Instant::now() + STEP;
    let mut failing = false;
    while !stop.load(Ordering::Relaxed) {
        if Instant::now() < next {
            std::thread::sleep(Duration::from_millis(250));
            continue;
        }
        live.push(&mic.read(), &computer.read());
        match live.step() {
            Ok(_) => {
                failing = false;
                next = Instant::now() + STEP;
            }
            Err(e) => {
                // Said once per outage; what is missed now is done at the end.
                if !failing {
                    eprintln!("{APP_NAME}: live transcription paused: {e}");
                }
                failing = true;
                next = Instant::now() + BACK_OFF;
            }
        }
    }
    let prior = live.prior();
    eprintln!(
        "{APP_NAME}: {:.0}s of the recording transcribed live",
        prior.seconds()
    );
    prior
}

// ---------------------------------------------------------------------------
// Command line

/// `transcribe-live <mic> <computer> [--language xx]`: plays two recorded
/// tracks through the live passes as if they were being recorded, then
/// transcribes what is left as at the end of a meeting. For trying the worker
/// on a real meeting without holding one.
pub fn cli(args: &[String]) -> glib::ExitCode {
    let mut paths = Vec::new();
    let mut language = "auto".to_owned();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--language" => language = args.next().cloned().unwrap_or(language),
            _ => paths.push(arg.clone()),
        }
    }
    let [mic, computer] = paths.as_slice() else {
        eprintln!("usage: {APP_NAME} transcribe-live <mic> <computer> [--language xx]");
        return glib::ExitCode::from(2);
    };
    let tracks = transcribe::load_track(Path::new(mic))
        .and_then(|m| Ok((m, transcribe::load_track(Path::new(computer))?)));
    let (mic, computer) = match tracks {
        Ok(tracks) => tracks,
        Err(e) => {
            eprintln!("{APP_NAME}: {e}");
            return glib::ExitCode::FAILURE;
        }
    };
    let remote = match Remote::connect() {
        Ok(Some(remote)) => remote,
        Ok(None) => {
            eprintln!("{APP_NAME}: no worker set (remote = ... in config.toml)");
            return glib::ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("{APP_NAME}: {e}");
            return glib::ExitCode::FAILURE;
        }
    };
    eprintln!("Live passes on {}", remote.url());
    let mut live = Live::new(Some(remote), &language);
    let step = STEP.as_secs() as usize * WHISPER_RATE;
    let len = mic.len().max(computer.len());
    let (mut busy, mut slowest) = (Duration::ZERO, Duration::ZERO);
    let started = Instant::now();
    let mut at = 0;
    while at < len {
        let to = (at + step).min(len);
        let part = |t: &[f32]| t[at.min(t.len())..to.min(t.len())].to_vec();
        live.push(&part(&mic), &part(&computer));
        let begun = Instant::now();
        if let Err(e) = live.step() {
            eprintln!("{APP_NAME}: {e}");
            return glib::ExitCode::FAILURE;
        }
        let took = begun.elapsed();
        busy += took;
        slowest = slowest.max(took);
        at = to;
    }
    let prior = live.prior();
    let recorded = len as f64 / WHISPER_RATE as f64;
    eprintln!(
        "Live: {:.0}s of {:.0}s recorded transcribed in {:.0}s of worker time \
         (slowest step {:.1}s, every step has {}s); {:.0}s elapsed",
        prior.seconds(),
        recorded,
        busy.as_secs_f64(),
        slowest.as_secs_f64(),
        STEP.as_secs(),
        started.elapsed().as_secs_f64(),
    );
    eprintln!("After the meeting:");
    transcribe::run_cli(|events, abort| {
        transcribe::transcribe(&mic, &computer, &language, Some(&prior), events, abort)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_resampler_matches_downsampling_all_at_once() {
        let input: Vec<f32> = (0..48_000).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        let whole = transcribe::downsample_for_test(&input, 3);
        let mut resampler = Resampler::new(3);
        let mut streamed = Vec::new();
        for chunk in input.chunks(4_801) {
            streamed.extend(resampler.feed(chunk));
        }
        // The stream holds back the last few samples until more audio comes.
        assert!(whole.len() - streamed.len() <= TAPS / 2 / 3 + 1);
        for (a, b) in whole.iter().zip(&streamed) {
            assert!((a - b).abs() < 1e-6, "{a} {b}");
        }
    }

    #[test]
    fn levels_come_in_whole_frames_and_survive_a_trim() {
        let mut track = Track::default();
        track.push(&vec![0.1; FRAME * 3 + 10]);
        assert_eq!(track.levels.len(), 3);
        track.trim(FRAME * 2);
        assert_eq!(track.base, FRAME * 2);
        track.push(&vec![0.1; FRAME]);
        assert_eq!(track.levels.len(), 4);
        assert!((track.levels[3] - 0.1).abs() < 1e-6);
    }
}
