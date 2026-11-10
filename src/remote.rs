//! Whisper on another machine: a worker (`omarchy-whisper-worker`) with a GPU
//! to spare runs the passes, so the laptop's own GPU stays free for the screen.
//! Set it in config.toml:
//!
//!   remote = "http://eve:8178"
//!
//! or for one run with `OMARCHY_MEETING_REMOTE` (empty turns it off). The
//! silence gate, the speakers and the lines stay here; only the audio of each
//! pass goes out, and only its words come back.

use std::time::Duration;

pub use whisper_words::Word;

/// How long a health check may take before the worker counts as away.
const REACH: Duration = Duration::from_secs(3);
/// A pass of ten minutes takes well under a minute on a worker; this is for
/// a worker that hangs, not a slow one.
const PASS: Duration = Duration::from_secs(600);

#[derive(Clone)]
pub struct Remote {
    url: String,
    agent: ureq::Agent,
}

/// The configured worker URL, if any.
pub fn configured() -> Option<String> {
    if let Some(value) = std::env::var_os("OMARCHY_MEETING_REMOTE") {
        let value = value.to_string_lossy().trim().to_owned();
        return (!value.is_empty()).then_some(value);
    }
    let text = std::fs::read_to_string(crate::models::config_file()).ok()?;
    text.lines()
        .find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == "remote").then(|| {
                let value = value.split('#').next().unwrap_or("");
                value.trim().trim_matches('"').trim_end_matches('/').to_owned()
            })
        })
        .filter(|url| !url.is_empty())
}

impl Remote {
    /// The configured worker, when it answers.
    pub fn connect() -> Result<Option<Remote>, String> {
        let Some(url) = configured() else {
            return Ok(None);
        };
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(PASS))
            .build()
            .into();
        let remote = Remote { url, agent };
        remote.health()?;
        Ok(Some(remote))
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    fn health(&self) -> Result<(), String> {
        let url = format!("{}/v1/health", self.url);
        let mut response = self
            .agent
            .get(&url)
            .config()
            .timeout_global(Some(REACH))
            .build()
            .call()
            .map_err(|e| format!("{}: {e}", self.url))?;
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|e| format!("{}: {e}", self.url))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", self.url))?;
        if value["ok"].as_bool() == Some(true) {
            Ok(())
        } else {
            Err(format!("{}: not ready", self.url))
        }
    }

    /// One whisper pass over `samples` (16 kHz mono): its words, with times in
    /// `samples`, and the language whisper used or found.
    pub fn words(
        &self,
        samples: &[f32],
        language: &str,
    ) -> Result<(Vec<Word>, Option<String>), String> {
        let url = format!("{}/v1/words?language={language}", self.url);
        let body = whisper_words::to_s16le(samples);
        let mut response = self
            .agent
            .post(&url)
            .header("Content-Type", "application/octet-stream")
            .send(&body[..])
            .map_err(|e| format!("{}: {e}", self.url))?;
        let text = response
            .body_mut()
            .with_config()
            .limit(64 << 20)
            .read_to_string()
            .map_err(|e| format!("{}: {e}", self.url))?;
        parse(&text).map_err(|e| format!("{}: {e}", self.url))
    }
}

fn parse(text: &str) -> Result<(Vec<Word>, Option<String>), String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let words = value["words"]
        .as_array()
        .ok_or("the answer has no words")?
        .iter()
        .map(|w| {
            Some(Word {
                text: w[0].as_str()?.to_owned(),
                start_ms: w[1].as_i64()?,
                end_ms: w[2].as_i64()?,
                no_speech: w[3].as_f64()? as f32,
                segment: w[4].as_u64()? as usize,
            })
        })
        .collect::<Option<Vec<Word>>>()
        .ok_or("a word in the answer is malformed")?;
    Ok((words, value["language"].as_str().map(str::to_owned)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_turns_back_into_words() {
        let (words, language) = parse(
            r#"{"language":"en","words":[["Hello",120,120,0.01,0],["there.",480,520,0.01,0]]}"#,
        )
        .unwrap();
        assert_eq!(language.as_deref(), Some("en"));
        assert_eq!(words.len(), 2);
        assert_eq!(words[1].text, "there.");
        assert_eq!(words[1].end_ms, 520);
    }

    #[test]
    fn a_malformed_word_is_an_error() {
        assert!(parse(r#"{"words":[["Hello","x",1,0.0,0]]}"#).is_err());
    }
}
