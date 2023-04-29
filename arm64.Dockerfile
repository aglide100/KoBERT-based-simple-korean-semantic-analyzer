from mxnet/python:1.9.1_aarch64_cpu_py3

WORKDIR /app
ENV JAVA_HOME /usr/lib/jvm/java-1.7-openjdk/jre
RUN apt-get update && apt-get install -y libssl-dev zlib1g-dev gcc g++ make git wget pkg-config default-jre

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN mkdir -p skt/kobert-base-v1 && cd skt/kobert-base-v1 && \ 
    wget https://huggingface.co/skt/kobert-base-v1/raw/main/tokenizer_config.json && \
    wget https://huggingface.co/skt/kobert-base-v1/resolve/main/spiece.model && \
    wget https://huggingface.co/skt/kobert-base-v1/raw/main/special_tokens_map.json && \
    cd ../..

COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN pip3 install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

RUN pip install quickspacer tensorflow_io==0.29.0 konlpy

# RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -s
COPY . .

RUN touch Test.py && cat Analyzer.py >> Test.py && cat TestCode.py >> Test.py && rm -f TestCode.py

ENTRYPOINT ["python3"]
