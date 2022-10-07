FROM nvcr.io/nvidia/pytorch:22.09-py3

RUN pip install -U pytorch-lightning[extra]
RUN pip install seqchromloader
RUN pip install seaborn
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y
RUN apt install bedtools
CMD jupyter lab --port=8889
