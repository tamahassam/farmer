docker run \
	--gpus all \
	-itd \
	--rm \
	-p $2:22 \
	--name $1 \
	--mount type=bind,source=/mnt,target=/mnt \
	--mount type=bind,source=/home,target=/home \
	test:v1 \
	fish

docker exec -it --user $USER $1 fish
