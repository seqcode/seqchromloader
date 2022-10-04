FROM nvcr.io/nvidia/pytorch:22.09-py3

RUN pip install -U pytorch-lightning[extra]
RUN pip install seqchromloader
