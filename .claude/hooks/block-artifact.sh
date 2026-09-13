#!/usr/bin/env bash
set -euo pipefail

jq -n '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: "Artifacts are disabled in this repo. Write to a file or the terminal."
  }
}'
