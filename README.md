# HFAR
Datasets, codes and some materials for paper 《**Learning Human Feedback from Large Language Models for Content Quality-aware Recommendation**》

## Datasets
MIND Dataset: MIcrosoft News Dataset Download Url: [https://msnews.github.io/](https://msnews.github.io/)

EB-NeRD Dataset: Download Url: [https://recsys.eb.dk/dataset/](https://recsys.eb.dk/dataset/)


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
Supplementary Information for “Learning Human Feedback
from Large Language Models for Content Quality-aware
Recommendation”.
> we collect some examples to verify that unethical recommended content in real systems have caused a lot of blame from users on social network platform.

(Warning: this document contains example data that may be offensive or harmful. )

 

