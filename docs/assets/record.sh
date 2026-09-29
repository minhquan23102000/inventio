#!/usr/bin/env bash
# Records demo.sh into demo.gif (brew install asciinema agg).
set -e
here="$(cd "$(dirname "$0")" && pwd)"
asciinema rec --headless --overwrite --window-size 102x19 -i 1.5 -c "bash $here/demo.sh" "$here/demo.cast"
agg --theme f5f4ed,141413,141413,9a3b2e,3f6b3a,8a6d1f,1b365d,6b3f6b,2d5a8a,6b6a64 --font-family Menlo --font-size 20 --line-height 1.3 --last-frame-duration 4 "$here/demo.cast" "$here/demo.gif"
rm "$here/demo.cast"
