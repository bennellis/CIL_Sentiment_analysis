Sentiment Classification of User Reviews
Authors
Ben Ellis, Bruce Balfour, Florian Zogaj, Jakob Hütteneder
Department of Computer Science, ETH Zurich

Overview
This project focuses on sentence-level sentiment classification of user-generated reviews. Each review is categorized into one of three sentiment classes: positive, neutral, or negative.

Using a dataset of over 100,000 labeled sentences, we explore a range of approaches — from traditional machine learning models using Bag-of-Words (BoW) to fine-tuned transformer-based architectures like BERT and DeBERTa. The goal is to not only maximize performance but also to use a custom metric that penalizes severe misclassifications more heavily.

Dataset
~102,000 single-sentence user reviews

Labels inferred from star ratings:

Positive: 30.5%

Neutral: 48%

Negative: 21.5%

Reviews contain informal language (URLs, typos, slang)

Mostly English (>99.5%) with some non-English samples

Challenges: sarcasm, mixed sentiments, noisy text

Preprocessing
We evaluated multiple preprocessing pipelines depending on the model type:

For Traditional Models
Lowercasing

Removing URLs, punctuation, and stopwords

Normalizing repeated characters

Lemmatization

Translating to English

For Transformer-Based Models
Light preprocessing:

Expanding contractions

Removing URLs

Normalizing repeated characters

Translating to English

Avoided lemmatization/stopword removal due to conflict with model pretraining

We also performed back-translation for data augmentation.

Models
Baselines
BoW + Logistic Regression

DistilBERT + MLP Head

Simple Feedforward Neural Network

Human Labeling Benchmark: ~87% accuracy on 500 samples

Transformer Models
DistilBERT

BERT (Base & Large)

RoBERTa (Base & Large)

ModernBERT (Base & Large)

DeBERTa-v3 (Base & Large)

Fine-tuning Strategies
Fully frozen encoder with trained head

Full fine-tuning of encoder + head (best performance)

Partial fine-tuning (less effective)

Best results were obtained using full fine-tuning of DeBERTa-v3, achieving a 92.0 validation score and 91.0 test score
After ensembling both DeBERTa-v3 base and DeBERTa-v3 large from different checkpoints and using various classification heads, we were able to achieve a test score of 91.5

Custom Loss Function
We used a custom scoring function (L) focused on penalizing large misclassification errors more heavily:

Score = 1.0 for correct predictions

Score = 0.5 for one-class-off predictions

Score = 0.0 for two-class-off predictions

To optimize this, we experimented with:

MAE-based soft loss

Hybrid MAE + Cross Entropy loss (for smoother gradients)

Class Distance Weighted Cross Entropy (CDW-CE) — our final choice

which balances meaningful gradients with stronger penalties for severe misclassifications.

Results & Observations
Data cleaning and preprocessing did not significantly impact model performance.

Full fine-tuning beats using a frozen encoder methods by ~4%.

CDW-CE outperforms basic MAE or CE in terms of both gradient stability and alignment with real-world cost of error.

Early stopping and validation monitoring are crucial to avoid overfitting.

Find our full report uploaded with our github.



The main directory used for testing our models was the Hyperparameters directory. 

To run use or code to train some models, pip install the requirements.txt using Python 3.10, and simply edit the config in Hyperparameters/config/config.yaml to the configuration you want, and run the main.py file.

If you wish to test multiple models in the ensemble, you will have to edit the lists located in the ensemble.py file to work with the weights you have saved.