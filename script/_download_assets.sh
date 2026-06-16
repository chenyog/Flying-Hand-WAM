#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
ASSETS_DIR="$PROJECT_ROOT/assets"

cd "$ASSETS_DIR"
python _download.py

required_zips=(
    embodiments.zip
    objects.zip
    background_texture.zip
)

for zip_file in "${required_zips[@]}"; do
    if [ ! -s "$zip_file" ]; then
        echo "Missing or empty file: $ASSETS_DIR/$zip_file" >&2
        exit 1
    fi
    echo "Checking $zip_file ..."
    unzip -t "$zip_file" >/dev/null
done

for zip_file in "${required_zips[@]}"; do
    echo "Extracting $zip_file ..."
    unzip -o "$zip_file"
done

rm -f "${required_zips[@]}"

cd "$PROJECT_ROOT"
echo "Configuring Path ..."
python ./script/update_embodiment_config_path.py
