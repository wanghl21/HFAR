# HFAR
Datasets, codes and some materials for paper 《**Learning Human Feedback from Large Language Models for Content Quality-aware Recommendation**》

## Datasets
MIND Dataset: [https://msnews.github.io/](https://msnews.github.io/)

EB-NeRD Dataset: [https://recsys.eb.dk/dataset/](https://recsys.eb.dk/dataset/)

## Codes

### Code function description
**metrics_utils.py**
> codes including the functions for computing AUC、MRR、nDCG@5 and nDCG@10, and a new metric named RQ@K to measure the morality of the recommender system based on the frequency of negative morality information in the top K recalled recommended items.

**model_utils.py**
> codes including the nerual networks for basic recommendation models.

**generator_utils.py**
> codes including the functions that process the dataset for model training.

**process_utils.py**
> codes including the functions that process the raw data like reading item content and parse user behaviors.

**rec.py**
> codes for training the quality-aware recommendation models.

**inference_rec.py**
> codes for quality-aware recommendation inference. 

### **Training**
```bash
./ run_rec.sh
```
### Inference
```bash
python inference_rec.py $method $beta
```

## Other Materials 
Supplementary Information for “Learning Human Feedback from Large Language Models for Content Quality-aware Recommendation”.

**Appendix.pdf**
> Details for experimental settings.
> We also collect some examples to verify that unethical recommended content in real systems have caused a lot of blame from users on social network platform.
>
> (Warning: this document contains example data that may be offensive or harmful. )

**docs_factors_sample.xlsx**
> Some examples of llm annotation results.




 

