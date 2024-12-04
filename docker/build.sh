set -e

# Resolve symlinks if there is any
find . -type l -exec sh -c 'cp -L "$0" "$0.tmp" && mv "$0.tmp" "$0"' {} \;

docker build --progress=plain . -f docker/Dockerfile -t modelopt_examples:latest "$@"

# restore symlinks if there is any
git status | grep 'typechange:' | awk '{print $2}' | xargs git checkout --
