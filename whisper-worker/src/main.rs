//! Runs the recorder's whisper passes for it, on a machine with a GPU to
//! spare. The recorder posts the audio of one pass (16 kHz mono s16le) and
//! gets the words back with their times; everything else, the silence gate,
//! the speakers and the lines, stays in the recorder.
//!
//!   omarchy-whisper-worker --model ~/models/ggml-large-v3-turbo.bin --listen 100.64.0.1:8178
//!
//! POST /v1/words?language=en   body: s16le samples   → {"language", "words": [[text, start_ms, end_ms, no_speech, segment]], ...}
//! GET  /v1/health                                    → {"ok": true, "model": ...}

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;
use tiny_http::{Header, Method, Request, Response, Server};
use whisper_rs::WhisperContext;

/// A longer pass than this is a mistake, not a meeting: about three hours.
const MAX_BODY: usize = 3 * 3600 * 16_000 * 2;

struct Worker {
    context: WhisperContext,
    model: String,
    /// One pass at a time: they share the GPU, and a second would only slow
    /// the first down.
    gpu: Mutex<()>,
}

fn main() {
    let mut model: Option<PathBuf> = None;
    let mut listen = "127.0.0.1:8178".to_owned();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next().map(PathBuf::from),
            "--listen" => listen = args.next().unwrap_or(listen),
            _ => {
                eprintln!("usage: omarchy-whisper-worker --model <ggml-*.bin> [--listen host:port]");
                std::process::exit(2);
            }
        }
    }
    let Some(model) = model else {
        eprintln!("omarchy-whisper-worker: --model is required");
        std::process::exit(2);
    };
    let name = model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    whisper_rs::install_logging_hooks();
    let context = match whisper_words::load(&model, whisper_words::preset_for(&name), true) {
        Ok(context) => context,
        Err(e) => {
            eprintln!("omarchy-whisper-worker: {e}");
            std::process::exit(1);
        }
    };
    let worker = Arc::new(Worker {
        context,
        model: name,
        gpu: Mutex::new(()),
    });
    let server = match Server::http(&listen) {
        Ok(server) => server,
        Err(e) => {
            eprintln!("omarchy-whisper-worker: cannot listen on {listen}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("omarchy-whisper-worker: {} on http://{listen}", worker.model);
    for request in server.incoming_requests() {
        let worker = worker.clone();
        std::thread::spawn(move || handle(&worker, request));
    }
}

fn handle(worker: &Worker, mut request: Request) {
    let url = request.url().to_owned();
    let (path, query) = url.split_once('?').unwrap_or((&url, ""));
    let response = match (request.method(), path) {
        (Method::Get, "/v1/health") => json_response(
            200,
            json!({ "ok": true, "model": worker.model, "busy": worker.gpu.try_lock().is_err() }),
        ),
        (Method::Post, "/v1/words") => {
            let language = query
                .split('&')
                .find_map(|pair| pair.strip_prefix("language="))
                .filter(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_alphabetic()))
                .unwrap_or("auto")
                .to_owned();
            let mut body = Vec::new();
            let read = std::io::Read::read_to_end(
                &mut std::io::Read::take(request.as_reader(), MAX_BODY as u64 + 1),
                &mut body,
            );
            match read {
                Err(e) => error(400, &format!("reading the audio: {e}")),
                Ok(_) if body.len() > MAX_BODY => error(413, "too much audio for one pass"),
                Ok(_) => match transcribe(worker, &body, &language) {
                    Ok(value) => json_response(200, value),
                    Err(e) => error(500, &e),
                },
            }
        }
        _ => error(404, "not found"),
    };
    let _ = request.respond(response);
}

fn transcribe(worker: &Worker, body: &[u8], language: &str) -> Result<serde_json::Value, String> {
    let samples = whisper_words::from_s16le(body);
    let audio_ms = samples.len() as i64 * 1000 / 16_000;
    let _gpu = worker.gpu.lock().unwrap_or_else(|e| e.into_inner());
    let started = Instant::now();
    let mut state = worker.context.create_state().map_err(|e| e.to_string())?;
    let params = whisper_words::params(language);
    state.full(params, &samples).map_err(|e| e.to_string())?;
    let (words, detected) = whisper_words::words(&worker.context, &state);
    let took_ms = started.elapsed().as_millis() as i64;
    eprintln!(
        "omarchy-whisper-worker: {:.1}s of audio in {:.1}s ({:.0}x), {} words",
        audio_ms as f64 / 1000.0,
        took_ms as f64 / 1000.0,
        audio_ms as f64 / took_ms.max(1) as f64,
        words.len()
    );
    let words: Vec<serde_json::Value> = words
        .into_iter()
        .map(|w| json!([w.text, w.start_ms, w.end_ms, w.no_speech, w.segment]))
        .collect();
    Ok(json!({
        "language": detected,
        "audio_ms": audio_ms,
        "took_ms": took_ms,
        "model": worker.model,
        "words": words,
    }))
}

fn json_response(status: u16, value: serde_json::Value) -> Response<std::io::Cursor<Vec<u8>>> {
    let header = Header::from_bytes("Content-Type", "application/json").expect("static header");
    Response::from_data(value.to_string().into_bytes())
        .with_status_code(status)
        .with_header(header)
}

fn error(status: u16, message: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(status, json!({ "ok": false, "error": message }))
}
