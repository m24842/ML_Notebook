#!/bin/bash

set -e

PYPROJECT="pyproject.toml"

if [ ! -f "$PYPROJECT" ]; then
    echo "pyproject.toml not found!"
    exit 1
fi

echo "Select upgrade type:"
select choice in "major" "minor" "bug"; do
    case $choice in
        major|minor|bug) break;;
        *) echo "Invalid choice";;
    esac
done

# Extract the current version from pyproject.toml
version_line=$(grep -E '^\s*version\s*=' "$PYPROJECT" | head -n 1)

if [[ $version_line =~ ([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    bug="${BASH_REMATCH[3]}"
else
    echo "Failed to parse version!"
    exit 1
fi

# Increment the appropriate part
case $choice in
    major)
        ((major++))
        minor=0
        bug=0
        ;;
    minor)
        ((minor++))
        bug=0
        ;;
    bug)
        ((bug++))
        ;;
esac

new_version="$major.$minor.$bug"
echo "Upgraded to version: $new_version"

# Replace the version line in pyproject.toml
# This handles cases where there may be leading whitespace or different spacing around =
sed -i.bak -E "s/^( *version *= *\")([0-9]+\.[0-9]+\.[0-9]+)(\")/\1$new_version\3/" "$PYPROJECT"

# Optional: Build and upload
python -m build
twine upload dist/*
