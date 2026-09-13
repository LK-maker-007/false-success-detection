#!/usr/bin/env bash
set -euo pipefail

payload=$(cat)
command=$(jq -r '.tool_input.command // ""' <<<"$payload")

case "$command" in
  *"git commit"*|*"gh pr create"*|*"gh pr edit"*|*"git tag"*) ;;
  *) exit 0 ;;
esac

forbidden='Co-[Aa]uthored-[Bb]y: *Claude|Generated with \[?Claude|claude\.ai/code|noreply@anthropic\.com|🤖'

if grep -qE "$forbidden" <<<"$command"; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "Attribution marker in commit/PR text. Remove it and retry."
    }
  }'
  exit 0
fi

exit 0
