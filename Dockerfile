FROM nvcr.io/nvidia/tritonserver:23.04-py3

COPY triton-docker-requirements.txt .
RUN pip3 install -r triton-docker-requirements.txt

ENTRYPOINT [ "tritonserver" ]
