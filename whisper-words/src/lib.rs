//! The whisper settings and the word timing, shared by the recorder and its
//! remote worker, so a transcript comes out the same wherever whisper runs.

use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters, WhisperState,
};

/// A word with its times in the audio whisper was given, and how sure whisper
/// was that its segment held speech at all.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub no_speech: f32,
    /// Index of the whisper segment it came from.
    pub segment: usize,
}

/// Loads a model. Word times are aligned on the attention heads (DTW) when
/// the model is of a known kind: the plain token times drift by up to a
/// second, too much to tell where one speaker takes over.
pub fn load(
    model: &std::path::Path,
    preset: Option<DtwModelPreset>,
    gpu: bool,
) -> Result<WhisperContext, String> {
    let mut context_params = WhisperContextParameters::default();
    context_params.use_gpu(gpu);
    if let Some(model_preset) = preset {
        context_params.dtw_parameters(DtwParameters {
            mode: DtwMode::ModelPreset { model_preset },
            ..Default::default()
        });
    }
    WhisperContext::new_with_params(model, context_params)
        .map_err(|e| format!("could not load the model {}: {e}", model.display()))
}

/// The parameters of every pass. Callbacks are the caller's to add.
pub fn params(language: &str) -> FullParams<'_, '_> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    params.set_n_threads(threads.min(16) as i32);
    params.set_language(Some(language));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_no_speech_thold(0.6);
    // Word times, so a segment can be split where the speaker changes.
    params.set_token_timestamps(true);
    params.set_split_on_word(true);
    params
}

/// The words of a finished pass, and the language whisper used or found.
pub fn words(context: &WhisperContext, state: &WhisperState) -> (Vec<Word>, Option<String>) {
    let detected = whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(str::to_owned);
    let eot = context.token_eot();
    let mut words: Vec<Word> = Vec::new();
    for (index, segment) in state.as_iter().enumerate() {
        let no_speech = segment.no_speech_probability();
        // Bytes first: a character can be split over two tokens.
        let mut current: Option<(Vec<u8>, i64, i64)> = None;
        let flush = |current: &mut Option<(Vec<u8>, i64, i64)>, words: &mut Vec<Word>| {
            if let Some((bytes, start_ms, end_ms)) = current.take() {
                let text = String::from_utf8_lossy(&bytes).trim().to_owned();
                if !text.is_empty() {
                    words.push(Word {
                        text,
                        start_ms,
                        end_ms,
                        no_speech,
                        segment: index,
                    });
                }
            }
        };
        for i in 0..segment.n_tokens() {
            let Some(token) = segment.get_token(i) else {
                continue;
            };
            if token.token_id() >= eot {
                continue; // timestamps and other special tokens
            }
            let Ok(bytes) = token.to_bytes() else {
                continue;
            };
            let data = token.token_data();
            // DTW gives one moment per token; fall back to the plain times.
            let (t0, t1) = if data.t_dtw >= 0 {
                (data.t_dtw * 10, data.t_dtw * 10)
            } else {
                (data.t0 * 10, data.t1 * 10)
            };
            match current.as_mut() {
                Some((word, _, end)) if !bytes.starts_with(b" ") => {
                    word.extend_from_slice(bytes);
                    *end = t1.max(*end);
                }
                _ => {
                    flush(&mut current, &mut words);
                    current = Some((bytes.to_vec(), t0, t1));
                }
            }
        }
        flush(&mut current, &mut words);
    }
    (words, detected)
}

/// 16 kHz mono f32 as the s16le bytes the worker takes.
pub fn to_s16le(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// The worker's side of `to_s16le`.
pub fn from_s16le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
        .collect()
}

/// The model preset for a model file name such as `ggml-large-v3-turbo.bin`.
pub fn preset_for(file_name: &str) -> Option<DtwModelPreset> {
    let name = file_name.strip_prefix("ggml-").unwrap_or(file_name);
    let name = name.strip_suffix(".bin").unwrap_or(name);
    Some(match name {
        "tiny" => DtwModelPreset::Tiny,
        "tiny.en" => DtwModelPreset::TinyEn,
        "base" => DtwModelPreset::Base,
        "base.en" => DtwModelPreset::BaseEn,
        "small" => DtwModelPreset::Small,
        "small.en" => DtwModelPreset::SmallEn,
        "medium" => DtwModelPreset::Medium,
        "medium.en" => DtwModelPreset::MediumEn,
        "large-v3" => DtwModelPreset::LargeV3,
        "large-v3-turbo" => DtwModelPreset::LargeV3Turbo,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_survive_the_wire() {
        let samples = [0.0, 0.5, -0.5, 0.999, -1.0];
        let back = from_s16le(&to_s16le(&samples));
        for (a, b) in samples.iter().zip(back) {
            assert!((a - b).abs() < 1e-4, "{a} {b}");
        }
    }

    #[test]
    fn presets_follow_the_file_name() {
        assert!(matches!(
            preset_for("ggml-large-v3-turbo.bin"),
            Some(DtwModelPreset::LargeV3Turbo)
        ));
        assert!(preset_for("my-model.bin").is_none());
    }
}
