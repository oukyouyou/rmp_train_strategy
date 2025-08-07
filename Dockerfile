FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV http_proxy=http://proxy.l2.med.tohoku.ac.jp:8080
ENV https_proxy=http://proxy.l2.med.tohoku.ac.jp:8080

RUN pip install matplotlib==3.7.2 \
  && pip install tensorboard