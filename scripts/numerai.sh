docker rm -f numerai-$1-$2-$3 || true
docker run -v /disk2/mw4315/numerai-classic/numerai-$1:/workspace/ --gpus device=$3 -it -d --rm --name numerai-$1-$2-$3 numerai:base
docker exec numerai-$1-$2-$3 python numerai-classic-$1-$2.py