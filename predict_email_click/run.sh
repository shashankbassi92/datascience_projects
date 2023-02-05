#!/bin/bash
set -e

IMAGE="email_click:latest"
SERVICE="email_click"
ID=$(id -u)


function start() {
#    docker run --name ML-env -p 8887:8888 -d -v predict_email_click:/home/jovyan/predict_email_click -e NB_UID=ID --user root nielsborie/ml-docker start-notebook.sh --NotebookApp.password="sha1:b6dba7097c97:7bded30fcbd5089adb3b63496d5e68921e102a5f"
    docker run --name ML-env --user root -d -p 8887:8888  -v predict_email_click:/home/jovyan/work/  nielsborie/ml-docker start-notebook.sh --NotebookApp.password="sha1:b6dba7097c97:7bded30fcbd5089adb3b63496d5e68921e102a5f"
    docker cp predict_email_click/data/data.parquet ML-env:/home/jovyan/work/.
#    docker start ML-env
    sleep 3
    docker ps
}

function stop() {
    docker stop ML-env
    docker rm ML-env
}

function build_image() {
    local current_uid=$1
    echo "Pulling ML image..."
    docker pull nielsborie/ml-docker
}

function print_usage() {
    cat <<EOF
Usage:

-- Options:

    `basename $0` build-image
        builds docker image with environment for running email click prediction algorithm
    `basename $0` start
        starts container for email click prediction
    `basename $0` restart
        restarts container
    `basename $0` stop
        stops service

EOF
}

#------------------------------------------------------------

case "$1" in
    "start")
        start
        ;;
    "stop")
        stop
        ;;
    "build-image")
        build_image
        ;;
    "restart")
        stop
        start
        ;;
    *)
        echo "Unknown option <$1>. Please tell me what to do :/"
        print_usage
        exit 1
        ;;
esac