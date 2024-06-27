# commands
docker run -it --gpus all --shm-size=8g -v $PWD:/code/ -v /home/gpss1/remote/datasets/Wildtrack_dataset:/data/Wildtrack -w /code mvdet


python main.py -d wildtrack
