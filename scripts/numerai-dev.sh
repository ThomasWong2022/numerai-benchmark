docker rm -f numerai-dev || true
docker run -v /disk2/mw4315/:/workspace/ --gpus device=$1 -it -p 8890:8890 -d --rm --name numerai-dev numerai:base
docker exec -d numerai-dev jupyter notebook --ip 0.0.0.0 --port 8890 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' &