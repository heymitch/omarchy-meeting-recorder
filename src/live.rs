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
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
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

impl SidePrior {
    /// Drops what the passes before held of `region`, to be sent again.
    fn forget(&mut self, region: &Region) {
        let inside = |ms: i64| {
            let at = ms.max(0) as usize * WHISPER_RATE / 1000;
            region.start <= at && at < region.end
        };
        self.words.retain(|w| !inside(w.start_ms));
        self.covered = self
            .covered
            .iter()
            .flat_map(|&(start, end)| {
                if end <= region.start || start >= region.end {
                    vec![(start, end)]
                } else {
                    [(start, region.start), (region.end, end)]
                        .into_iter()
                        .filter(|(a, b)| a < b)
                        .collect()
                }
            })
            .collect();
    }
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
/// At most this much in one pass, so a pass under way when the recording
/// stops is done in seconds; the rest goes in the next.
const MOST: usize = WHISPER_RATE * 60;
/// After a failed pass, wait this long before the next try.
const BACK_OFF: Duration = Duration::from_secs(30);

/// One side's audio so far, 16 kHz mono. All of it is kept: a region from
/// early on can still move and be sent again.
#[derive(Default)]
struct Track {
    /// As 16-bit, like the recording itself: half the memory of f32.
    samples: Vec<i16>,
    /// The samples after the last whole frame, as they came.
    tail: Vec<f32>,
    /// The level of every whole 30 ms frame since the start.
    levels: Vec<f32>,
    peak: f32,
}

impl Track {
    fn push(&mut self, new: &[f32]) {
        self.peak = new.iter().fold(self.peak, |m, s| m.max(s.abs()));
        self.samples
            .extend(new.iter().map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16));
        self.tail.extend_from_slice(new);
        let whole = self.tail.len() / FRAME * FRAME;
        self.levels
            .extend(self.tail[..whole].chunks_exact(FRAME).map(transcribe::rms));
        self.tail.drain(..whole);
    }

    /// The sound of `region` as the mix after the meeting has it.
    fn sound(&self, region: &Region, gain: f32) -> Vec<f32> {
        self.samples[region.start..region.end]
            .iter()
            .map(|s| transcribe::soft_clip(from_i16(*s) * gain))
            .collect()
    }
}

fn from_i16(sample: i16) -> f32 {
    f32::from(sample) / 32767.0
}

#[derive(Default)]
struct Side {
    track: Track,
    prior: SidePrior,
    /// Segment numbers given out so far, so every pass gets its own.
    segments: usize,
}

/// The live passes over a recording that keeps growing.
pub struct Live {
    remote: Option<Remote>,
    /// Finds and records the regions without sending them: `--dry`.
    dry: bool,
    language: String,
    mic: Side,
    computer: Side,
}

impl Live {
    pub fn new(remote: Option<Remote>, language: &str) -> Self {
        Live {
            remote,
            dry: false,
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

        if self.remote.is_none() && !self.dry {
            self.remote = Remote::connect()?;
        }
        let remote = self.remote.clone();
        if remote.is_none() && !self.dry {
            return Err("no worker is set".into());
        }
        let mut seconds = 0.0;
        for (side, regions, gain) in [
            (&mut self.mic, mic_regions, gains.0),
            (&mut self.computer, computer_regions, gains.1),
        ] {
            seconds += pass(remote.as_ref(), &mut self.language, side, regions, gain, len)?;
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

/// One side's pass over its finished regions that the passes before did not
/// hold, when there are enough of them. As the recording grows its noise
/// floor settles and a region can move; a moved region is sent again and its
/// new words replace the old ones, so by the end the passes hold the regions
/// as they are found over the whole recording.
fn pass(
    remote: Option<&Remote>,
    language: &mut String,
    side: &mut Side,
    regions: Vec<Region>,
    gain: f32,
    len: usize,
) -> Result<f64, String> {
    let joined = transcribe::joined_spans(&side.prior.covered);
    let ready: Vec<Region> = regions
        .into_iter()
        .filter(|r| {
            r.end + SETTLED <= len
                && r.end <= side.track.samples.len()
                && !transcribe::covered_by(&joined, r)
        })
        .collect();
    let Some(first) = ready.first() else {
        return Ok(0.0);
    };
    let waiting: usize = ready.iter().map(|r| r.end - r.start).sum();
    if waiting < BATCH && len - first.end < WAIT {
        return Ok(0.0);
    }
    let mut taken = 0;
    let ready: Vec<Region> = ready
        .into_iter()
        .take_while(|r| {
            let first = taken == 0;
            taken += r.end - r.start;
            first || taken <= MOST
        })
        .collect();
    let waiting: usize = ready.iter().map(|r| r.end - r.start).sum();
    let glued = Glued::glue(&ready, |r| side.track.sound(r, gain));
    let (words, found) = match remote {
        Some(remote) => remote.words(&glued.samples, language)?,
        None => (Vec::new(), None),
    };
    if *language == "auto"
        && let Some(found) = found
    {
        *language = found;
    }
    for region in &ready {
        side.prior.forget(region);
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
    /// What the passes so far transcribed, as of the last one that finished.
    done: Arc<Mutex<Prior>>,
}

/// Starts following the recording in `staging`, when a worker is set.
pub fn spawn(staging: &Path, language: &str) -> Option<Handle> {
    crate::remote::configured()?;
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(Mutex::new(Prior::default()));
    let (flag, out) = (stop.clone(), done.clone());
    let (staging, language) = (staging.to_owned(), language.to_owned());
    std::thread::Builder::new()
        .name("live-transcription".into())
        .spawn(move || follow(&staging, &language, &flag, &out))
        .ok()?;
    Some(Handle { stop, done })
}

impl Handle {
    /// Stops following and returns what the finished passes transcribed,
    /// right away: a pass still under way is left to end on its own, and its
    /// regions are transcribed with the rest after the meeting.
    pub fn finish(self) -> Prior {
        self.stop.store(true, Ordering::Relaxed);
        let prior = self.done.lock().unwrap_or_else(|e| e.into_inner()).clone();
        eprintln!(
            "{APP_NAME}: {:.0}s of the recording transcribed live",
            prior.seconds()
        );
        prior
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn follow(staging: &Path, language: &str, stop: &AtomicBool, done: &Mutex<Prior>) {
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
        let result = live.step();
        // A stop that came during the pass already took the last snapshot.
        if stop.load(Ordering::Relaxed) {
            break;
        }
        *done.lock().unwrap_or_else(|e| e.into_inner()) = live.prior();
        match result {
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
    let mut dry = false;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--language" => language = args.next().cloned().unwrap_or(language),
            "--dry" => dry = true,
            _ => paths.push(arg.clone()),
        }
    }
    let [mic, computer] = paths.as_slice() else {
        eprintln!("usage: {APP_NAME} transcribe-live <mic> <computer> [--language xx] [--dry]");
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
        _ if dry => None,
        Ok(Some(remote)) => Some(remote),
        Ok(None) => {
            eprintln!("{APP_NAME}: no worker set (remote = ... in config.toml)");
            return glib::ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("{APP_NAME}: {e}");
            return glib::ExitCode::FAILURE;
        }
    };
    if let Some(remote) = &remote {
        eprintln!("Live passes on {}", remote.url());
    }
    let mut live = Live::new(remote, &language);
    live.dry = dry;
    let step = STEP.as_secs() as usize * WHISPER_RATE;
    let len = mic.len().max(computer.len());
    let (mut busy, mut slowest) = (Duration::ZERO, Duration::ZERO);
    let mut sent = 0.0;
    let started = Instant::now();
    let mut at = 0;
    while at < len {
        let to = (at + step).min(len);
        let part = |t: &[f32]| t[at.min(t.len())..to.min(t.len())].to_vec();
        live.push(&part(&mic), &part(&computer));
        let begun = Instant::now();
        match live.step() {
            Ok(seconds) => sent += seconds,
            Err(e) => {
                eprintln!("{APP_NAME}: {e}");
                return glib::ExitCode::FAILURE;
            }
        }
        let took = begun.elapsed();
        busy += took;
        slowest = slowest.max(took);
        at = to;
    }
    let prior = live.prior();
    let recorded = len as f64 / WHISPER_RATE as f64;
    eprintln!(
        "Live: {:.0}s of speech sent ({:.0}s of it still held at the end) for {:.0}s \
         recorded, in {:.0}s of worker time (slowest step {:.1}s, every step has {}s); \
         {:.0}s elapsed",
        sent,
        prior.seconds(),
        recorded,
        busy.as_secs_f64(),
        slowest.as_secs_f64(),
        STEP.as_secs(),
        started.elapsed().as_secs_f64(),
    );
    eprintln!("Coverage: {}", transcribe::coverage(&mic, &computer, &prior));
    if dry {
        return glib::ExitCode::SUCCESS;
    }
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
    fn levels_come_in_whole_frames_whatever_the_pieces() {
        let mut whole = Track::default();
        whole.push(&vec![0.1; FRAME * 4]);
        let mut pieces = Track::default();
        for piece in vec![0.1; FRAME * 4].chunks(FRAME / 3 + 7) {
            pieces.push(piece);
        }
        assert_eq!(pieces.levels.len(), 4);
        assert_eq!(pieces.levels, whole.levels);
        assert_eq!(pieces.samples.len(), FRAME * 4);
    }
}
