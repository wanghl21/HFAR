# MoralRec
Datasets, codes and some materials for paper 《**Learning Human Feedback from Large Language Models for Content Quality-aware Recommendation**》

## Datasets
MIND Dataset: MIcrosoft News Dataset Download Url: [https://msnews.github.io/](https://msnews.github.io/)

The data attribute and content quality attribute of MIND-Small and MIND-Large dataset:

|  | **Data Attribute** |  |  | **Content  Quality Attribute** |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | #Impressions | #Clicks | #News | #Clickbait headlines | #Racial discrimination | #Violence |
| MIND-small | 230,117 | 347,727 | 65,238 | 1937 | 729 | 11551 |
| MIND-large | 15,777,377 | 24,155,470 | 104,151 | 1950 | 985 | 17560 |


## Codes
### Setup for codes
python==3.8
pytorch==1.12
scikit-learn==1.3.1
transformers==4.29.1

### Code function description
**metrics_utils.py: **
> codes including the functions for computing AUC、MRR、nDCG@5 and nDCG@10, and a new metric named RQ@K to measure the morality of the recommender system based on the frequency of negative morality information in the top K recalled recommended items.

**model_utils.py: **
> codes including the nerual networks for basic recommendation models.

**generator_utils.py:**
> codes including the functions that process the dataset for model training.

**process_utils.py:**
> codes including the functions that process the raw data like reading item content and parse user behaviors.

**rec.py:**
> codes for training the quality-aware recommendation models.

**inference_rec.py:**
> codes for quality-aware recommendation inference. 


### **Training **
```bash
./ run_rec.sh
```
### Inference
```bash
python inference_rec.py $method $beta
```

## Other Materials
《Some collected examples that people blame for negative quality recommended content in recommendation platforms.》
> we collect some examples to verify that unethical recommended content in real systems have caused a lot of blame from users on social network platform.

(Warning: this document contains example data that may be offensive or harmful. )

 

