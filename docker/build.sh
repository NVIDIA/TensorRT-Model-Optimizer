set -e

docker build . -f docker/Dockerfile -t modelopt_examples:latest "$@"
