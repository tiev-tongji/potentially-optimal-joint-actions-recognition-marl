#!/bin/bash
# Install StarCraft II and add the bundled SMAC maps.

set -e

POWQMIX_CODE_DIR=$(pwd)
export POWQMIX_CODE_DIR

echo 'Installing StarCraft II...'
cd "$HOME"
export SC2PATH="$HOME/StarCraftII"
echo "SC2PATH is set to $SC2PATH"

if [ -d "$SC2PATH" ]; then
    echo 'StarCraft II is already installed. Reinstall it? (y/n)'
    read -r response
    if [ "$response" = "y" ]; then
        echo 'Removing existing StarCraft II installation...'
        rm -rf "$SC2PATH"
        echo 'Existing installation removed.'
    else
        echo 'Skipping reinstall and continuing.'
    fi
fi

if [ ! -d "$SC2PATH" ]; then
    echo 'StarCraft II is not installed. Downloading SC2.4.10...'
    wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    unzip -P iagreetotheeula SC2.4.10.zip
else
    echo 'StarCraft II is already installed.'
fi

MAP_DIR="$SC2PATH/Maps/"

if [ ! -d "$MAP_DIR/SMAC_Maps" ]; then
    echo "MAP_DIR is set to $MAP_DIR"
    if [ ! -d "$MAP_DIR" ]; then
        mkdir -p "$MAP_DIR"
    fi

    cp -r "$POWQMIX_CODE_DIR/src/envs/smac_v2/official/maps/SMAC_Maps" "$MAP_DIR"
else
    echo 'SMAC v1/v2 maps are already installed.'
fi

echo 'StarCraft II and SMAC maps are installed.'
