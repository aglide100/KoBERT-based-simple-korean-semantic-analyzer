from continuumio/miniconda3

COPY . .

SHELL ["/bin/bash", "--login", "-c"]

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN conda env create -f environment.yml
RUN conda init bash

RUN conda activate myenv

RUN git clone https://github.com/SKTBrain/KoBERT.git
RUN cd KoBERT && rm requirements.txt && cp ../requirements.txt . && pip install .
# # RUN pip install opencv-python
RUN pip install -r requirements.txt

# RUN pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
ENTRYPOINT ["python3"]
