# CVML

## ML

- nearest neighbors: `knn.txt`
- k-means: `kmeans.py`

## Recommendation

- user-based recommendation: `user_based_rs.py`
- item-based recommendation: `item_based_rs.py`

    | Similarity | MAE | 
    | --- | --- |
    | MinMaxNormalization + Cosine | 0.29 |
    | ZScoreNormalization + Cosine | 0.40 |
    | MeanCenterNormalization + Cosine | 0.51 |
    | MinMaxNormalization + CosineSimilarity + UserSimilarityFactorization | 0.36 |
    | Cosine | 1.4 |
    | Cosine w. Item Similarity Factorization | 1.8 |
    | Pearson Correlation | 1.9 |
    | Mean Squared Difference | 2.6 |

- glslim alg.: `glslim.py`

## NLP
- huggingface transformers: read the book `Natural Language Processing with
  Transformers` and have familiarized myself with Hugging Face transformers
  - `huggingface-hello-transformer.py`
  - `huggingface-text-classification.ipynb`
  - `huggingface-transformer-anatomy.ipynb`
  - `huggingface-making-transformers-efficient-in-production.ipynb`
  - `huggingface-dealing-with-few-to-no-labels.ipynb`

---
## XGBoost
Quickly finished 1 round of `hands on gradient boosting with XGBoost and 
Scikit-learn`
