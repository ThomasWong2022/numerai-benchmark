docker rm -f numerai-signals-$1-$2 || true
docker run -v /disk2/mw4315/numerai-classic/numerai-signals:/workspace/ --gpus device=$2 -it -d --rm --name numerai-signals-$1-$2 numerai:base
docker exec numerai-signals-$1-$2 python numerai-signals-$1.py