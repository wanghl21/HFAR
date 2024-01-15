# QuaRec
Codes and Datasets For QuaRec Framework.
## Requirements for Codes

Environment:
python>=3.8
pytorch==1.12
scikit-learn==1.3.1
transformers==4.29.1
## Datasets
MIND: MIcrosoft News Dataset Download Url: [https://msnews.github.io/](https://msnews.github.io/)
The data attribute and content quality attribute of MIND-Small and MIND-Large dataset:

|  | Data Attribute |  |  | News Attribute |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Impressions | Clicks | News | clickbait headlines | racial discrimination | violence |
| MIND-small | 230,117 | 347,727 | 65,238 | 1937 | 729 | 11551 |
| MIND-large | 15,777,377 | 24,155,470 | 104,151 | 1950 | 985 | 17560 |


## Training
Download the data and get the quality datasets with LLMs. 
```bash
./ run_qua_alpha_method.sh
```
## Inference
```bash
python inference_qua_rec_classifier_contrastive_learning_qua.py method beta
```
