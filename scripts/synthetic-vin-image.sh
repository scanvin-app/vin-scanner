#!/usr/bin/env bash
# Render a synthetic 640x640 VIN plate image with ffmpeg (no photo needed).
# Usage: synthetic-vin-image.sh [out.png] [VIN]
# Env:   VIN_SCANNER_FONT — path to a TTF to use instead of fontconfig lookup.
set -euo pipefail

OUT="${1:-synthetic-vin.png}"
# Synthetic VIN with a valid ISO 3779 check digit (position 9).
VIN="${2:-WVWZZZ1K68W123456}"

if [[ -n "${VIN_SCANNER_FONT:-}" ]]; then
  FONT="fontfile=${VIN_SCANNER_FONT}"
else
  FONT="font=monospace"
fi

ffmpeg -v error -f lavfi -i "color=c=#d8d8d8:s=640x640:d=1" \
  -vf "drawbox=x=40:y=270:w=560:h=100:color=white:t=fill,drawtext=${FONT}:text='${VIN}':fontsize=52:fontcolor=#202020:x=(w-tw)/2:y=(h-th)/2" \
  -frames:v 1 -y "${OUT}"
echo "${VIN}"
