docker run --rm -it --user 1000:1002 --runtime=nvidia --name=llsrp-dev\
    -v <path-to-where-you-cloned-the-repo>:/opt/project \
    pytorch-image bash
