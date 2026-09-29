#!/usr/bin/env bash
# The README demo, typed and run for real. Record it with docs/assets/record.sh.
# Needs inventio with [dispositio] and a warm model server (any earlier query starts one).
cd "$(git rev-parse --show-toplevel)"
export INVENTIO_DB=/tmp/inv/map.db; rm -rf /tmp/inv

type_run() {
  printf '\033[1;34m$\033[0m '
  local s="$1"
  for ((i = 0; i < ${#s}; i++)); do printf '%s' "${s:i:1}"; sleep 0.03; done
  sleep 0.4; printf '\n'
  eval "$s" 2>/dev/null
  sleep "${2:-1.2}"
}

# indexed before the take (0.0 s each); the take starts at the question
inventio init examples/webshop/app --name app --public >/dev/null 2>&1
inventio init examples/webshop/wiki --name wiki --public >/dev/null 2>&1
clear
type_run 'inventio query "the nightly backup has not finished, what do I do?" -k 1' 3
type_run 'inventio read wiki:runbook.md:8-13' 3.5
