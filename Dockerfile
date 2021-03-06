FROM tensorflow/tensorflow
MAINTAINER tobe tobeg3oogle@gmail.com

ADD ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ADD . /model-player/
WORKDIR /model-player/

CMD ./predict_http_service.py
