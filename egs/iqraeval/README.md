text
# IqraEval Recipe

**IqraEval** is an Arabic phone recognition recipe for the [IqraEval Challenge](https://huggingface.co/IqraEval) using SSL features in Kaldi pipelines. 

### Download CV-Ar and TTS data from huggingface as described in [Iqra-Eval](https://github.com/Iqra-Eval/interspeech_IqraEval)

```
python download_hugg_data.py --path "IqraEval/Iqra_train" --split "train" --output_dir "./sws_data/CV-Ar"

python download_hugg_data.py --path "IqraEval/Iqra_train" --split "dev" --output_dir "./sws_data/CV-Ar"

python download_hugg_data_tts.py --path "IqraEval/Iqra_TTS" --split "train" --output_dir "./data/TTS" --dev_name "Amer"
```

 

## Expected Results (PER)
- 30-dimensional PCA for HMM-GMM systems
- SSL features are extracted at 50 fps (100 fps for MFCCs)
- A trigram language model is trained using KenLM for decoding
- Layer 9 performs best for the GMM system:

| SSL Layer | mono  | Δ+ΔΔ  | LDA+MLLT  |
| --------- | ----- | ----- | ----- |
| 5         | 31.96 | 24.27 | 22.63 |
| 6         | 29.31 | 22.29 | 20.93 |
| 7         | 28.17 | 22.27 | 20.69 |
| 8         | 27.18 | 20.72 | 19.41 |
| 9         | **26.59** | **20.48** | **18.82** |
| 10        | 27.08 | 20.67 | 19.13 |
| 11        | 29.49 | 22.62 | 21.57 |
| 12        | 30.11 | 22.84 | 21.90 |

- The following results show:
  - Comparison with MFCCs
  -  tdnnf system:
     -   raw SSL w/o ivectors
     -   reduced frame-subsampling-factor from the default 3 → 1 
     -   monophone topology    

| Model Type | mono  | Δ+ΔΔ  | LDA+MLLT  | tdnnf       |
| ---------- | ----- | ----- | ----- | ----------- |
| MFCC       | 53.85 | 43.47 | 41.65 | -           |
| SSL (9th layer)       | **26.59** | **20.48** | **18.82** |**11.56**       |
| IqraEval baseline       | - | - | - |16.42       |



## Citation
```
@misc{ssl_kaldi_iqraeval,
author = {Ibrahim Almajai},
title = {ssl-kaldi: IqraEval Recipe - Arabic Phone Recognition with mHuBERT},
year = {2025},
howpublished = {\url{https://github.com/ialmajai/ssl-kaldi/tree/main/egs/iqraeval}}
note = {Accessed: 2025-11}
}

@inproceedings{elkheir2025iqraeval, title = {Iqra’Eval: A Shared Task on Qur’anic Pronunciation Assessment},
author = {El Kheir, Yassine and Meghanani, Amit and Toyin, Hawau Olamide and Almarwani, Nada and Ibrahim, Omnia and Elshahawy, Youssef and Shahin, Mostafa and Ali, Ahmed},
booktitle = {Proceedings of the Third Arabic Natural Language Processing Conference}, 
year = {2025},
publisher = {Association for Computational Linguistics}, 
}


```
