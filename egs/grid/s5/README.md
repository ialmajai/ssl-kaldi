# Lipreading Recipe using AV-Hubert and Grid Corpus


## AV-HuBERT Setup

```
conda activate ssl-kaldi
pip install pip==24.0
pip install numpy==1.19.5
pip install -r requirements.txt
git clone https://github.com/facebookresearch/av_hubert.git --depth=1
cd av_hubert 
git submodule init
git submodule update
cd fairseq
pip install --editable ./ 
conda install -c conda-forge dlib==19.18.0
```
