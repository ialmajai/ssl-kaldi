# Lipreading Recipe using AV-Hubert and Grid Corpus


## AV-HuBERT Setup

```
conda activate ssl-kaldi
git clone https://github.com/facebookresearch/av_hubert.git && \
cd av_hubert && \
git submodule init && \
git submodule update && \
pip install -r requirements.txt && \
cd fairseq && \
pip install --editable ./ && \
cd .. && \
conda install -c conda-forge dlib==19.18.0
```
