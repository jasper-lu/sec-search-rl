#!/usr/bin/env bash
# Interactively fill in .env. Keys are typed without echo and never leave this
# machine; press Enter to keep an existing value or to leave an optional key blank.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE=.env
KEYS="TINKER_API_KEY WANDB_API_KEY OPENROUTER_API_KEY MIXEDBREAD_API_KEY"

note() {
  case "$1" in
    TINKER_API_KEY) echo "required for train and eval" ;;
    WANDB_API_KEY) echo "optional, only with --wandb-project" ;;
    OPENROUTER_API_KEY) echo "optional, only for api-eval --provider openrouter" ;;
    MIXEDBREAD_API_KEY) echo "optional, only for api-eval --provider mixedbread" ;;
  esac
}

existing() {
  [ -f "$ENV_FILE" ] && grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2- || true
}

echo "Writing $ENV_FILE."
{
  echo "# Load with: set -a; . ./.env; set +a"
  for key in $KEYS; do
    current=$(existing "$key")
    if [ -n "$current" ]; then prompt="$key ($(note "$key"); already set, Enter keeps it): "
    else prompt="$key ($(note "$key")): "; fi
    read -rs -p "$prompt" value </dev/tty; echo >&2
    echo "$key=${value:-$current}"
  done
} > "$ENV_FILE.tmp"
chmod 600 "$ENV_FILE.tmp"
mv "$ENV_FILE.tmp" "$ENV_FILE"
echo "Saved $ENV_FILE (mode 600)."
