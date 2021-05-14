docker exec -it farmer_tanaka bash -c \
    "cd $PWD && \
    poetry run python setup.py develop && \
    echo $PWD, $(date), $(id -u -n) >> ~/.farmerpath.csv"
