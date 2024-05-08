set -e

docker build --progress=plain . -f docker/Dockerfile -t modelopt_examples:latest "$@"
