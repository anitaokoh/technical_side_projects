FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
RUN export DOCKER_BUILDKIT=0
RUN export COMPOSE_DOCKER_CLI_BUILD=0
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
WORKDIR /app
COPY . /app
RUN pip3 --timeout=1000 --no-cache-dir install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install --timeout=500 --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD streamlit run main_app.py