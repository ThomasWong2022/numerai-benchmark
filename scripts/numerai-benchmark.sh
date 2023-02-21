docker rm -f numerai-signals-benchmark-$1-$2 || true
docker run -v /disk2/mw4315/numerai-classic/numerai-signals-benchmark:/workspace/ --gpus device=$2 -it -d --rm --name numerai-signals-benchmark-$1-$2 numerai:base
docker exec numerai-signals-benchmark-$1-$2 python numerai-signals-benchmark-$1.py