#!/bin/sh
# Builds the worker with Metal and runs it as a launchd agent on this Mac.
#   ./install.sh 100.64.0.1:8178     (the address to listen on, e.g. its Tailscale IP)
# Needs Rust and cmake, and the model in ~/.local/share/omarchy-meeting-recorder/models/.
set -eu
listen=${1:?usage: install.sh <host:port>}
here=$(cd "$(dirname "$0")" && pwd)
(cd "$here/.." && cargo build --release --features metal)
mkdir -p "$HOME/.local/bin" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cp "$here/../target/release/omarchy-whisper-worker" "$HOME/.local/bin/"
plist="$HOME/Library/LaunchAgents/ai.omarchy.whisper-worker.plist"
sed -e "s|@HOME@|$HOME|g" -e "s|@LISTEN@|$listen|g" "$here/omarchy-whisper-worker.plist.in" > "$plist"
launchctl bootout "gui/$(id -u)/ai.omarchy.whisper-worker" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$plist"
echo "omarchy-whisper-worker: listening on http://$listen (log: ~/Library/Logs/omarchy-whisper-worker.log)"
