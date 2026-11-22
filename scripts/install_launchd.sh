#!/bin/bash
# Installs two macOS LaunchAgents so training keeps going without supervision, across crashes and reboots:
#   com.autolearn.kaggle-campaign : the Kaggle campaign driver (restarted if it ever exits with an error)
#   com.autolearn.mac-watchdog    : resumes the local long training run if it dies or hangs
# Usage: scripts/install_launchd.sh <kaggle-username>      Remove: scripts/install_launchd.sh --uninstall
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
mkdir -p "$AGENTS" "$ROOT/logs"

unload() { launchctl bootout "$DOMAIN/$1" 2>/dev/null || true; }

if [[ "${1:-}" == "--uninstall" ]]; then
  for l in com.autolearn.kaggle-campaign com.autolearn.mac-watchdog; do unload "$l"; rm -f "$AGENTS/$l.plist"; done
  echo "removed"; exit 0
fi
USER_KAGGLE="${1:?usage: $0 <kaggle-username>}"

write_agent() {  # label, log file, args...
  local label="$1" logf="$2"; shift 2
  local args=""
  for x in "$@"; do args+="    <string>$x</string>"$'\n'; done
  cat > "$AGENTS/$label.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array>
$args  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>StandardOutPath</key><string>$logf</string>
  <key>StandardErrorPath</key><string>$logf</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>300</integer>
  <key>ProcessType</key><string>Background</string>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONUNBUFFERED</key><string>1</string></dict>
</dict>
</plist>
PLIST
  unload "$label"
  launchctl bootstrap "$DOMAIN" "$AGENTS/$label.plist"
  echo "installed $label"
}

write_agent com.autolearn.kaggle-campaign "$ROOT/logs/kaggle_campaign.log" \
  "$PY" "$ROOT/scripts/kaggle/campaign.py" --user "$USER_KAGGLE"
write_agent com.autolearn.mac-watchdog "$ROOT/logs/mac_watchdog.log" "$PY" "$ROOT/scripts/mac_watchdog.py"
