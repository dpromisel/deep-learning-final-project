# deep-learning-final-project

Sentiment Analysis on Amazon Customer Reviews

## How to use

python3 run.py
Args:
"sample/full": runs on sample or full dataset (default is full)
"lstm/transformer": runs with bidirectional LSTM model or Transformer model (default is LSTM)

## Introduction

Two broad areas of research focusing on how to learn useful representations: supervised and unsupervised learning. Supervised learning, or the training of high-capacity models on large labeled datasets, is an integral aspect of deep learning, particularly with respect to image recognition and machine translation. Unsupervised learning, on the other hand, is becoming more prominent in the field because of its scalability on datasets that cannot easily be cleaned or labeled. However, despite recent advances including pre training a model on a dataset and then fine tuning it for a given task, purely unsupervised approaches still exhibit weaker performance than supervised models. The authors of this paper hypothesize that poor corpus sources (usually novels) combined with limited capacity of current models lead to representational underfitting and lossy representations, especially when conducting sentiment analysis of reviews of consumer goods, as reviews and novels have small corpus overlaps.
Thus, this paper focuses on sentiment analysis and attempts to learn an unsupervised representation on a corpus of Amazon reviews. They do so with byte / character level language modelling due to its simplicity and generality, as well as to gauge if a low-level training objective can support high-level representations learning.
One reason we chose this paper is that the paper outlines clear objectives and results (giving us a good starting point as well as metrics for success). However, the paper concludes with the idea that “the underlying phenomena [that led to such an accurate model] remain more mysterious than clear.” This intrigued us, as there is still a great deal to be discovered in sentiment analysis in conjunction with unsupervised learning.
Because of limits on processing power and the fact that our data has clear labels, we are going to implement a supervised model.
Related Work
Given that we are comparing our version of supervised learning to their unsupervised sentiment analysis, we are using the existing paper and its results as our main source of inspiration and measure of success. Amazon reviews are also a widely used corpus source (used in many fields beyond DL), so we wanted to examine the results using DL with the potential to compare them to those from other areas of research.
Data
A collection of reviews written in the Amazon.com marketplace and associated metadata from 1995 until 2015. This is intended to facilitate study into the properties (and the evolution) of customer reviews potentially including how people evaluate and express their experiences with respect to products at scale. (130M+ customer reviews)

Each review comes with the following values:

Marketplace
2 letter country code of the marketplace where the review was written.
Customer_id
Random identifier that can be used to aggregate reviews written by a single author.
Review_id
The unique ID of the review.
Product_id
The unique Product ID the review pertains to
Product_parent
Random identifier that can be used to aggregate reviews for the same product.
Product_title
Title of the product.
Product_category
Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
Star_rating
The 1-5 star rating of the review.
Helpful_votes
Number of helpful votes.
Total_votes
Number of total votes the review received.
Vine
Review was written as part of the Vine program.
Verified_purchase
The review is on a verified purchase.
Review_headline
The title of the review.
Review_body
The review text.
Review_date
The date the review was written.

The dataset is currently available in two file formats.
Tab separated value (TSV), a text format - s3://amazon-reviews-pds/tsv/
Parquet, an optimized columnar binary format - s3://amazon-reviews-pds/parquet/
If you use the AWS Command Line Interface, you can list data in the bucket with the “ls” command:
aws s3 ls s3://amazon-reviews-pds/tsv/
To download data using the AWS Command Line Interface, you can use the “cp” command. For instance, the following command will copy the file named amazon_reviews_us_Camera_v1_00.tsv.gz to your local directory:
aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Camera_v1_00.tsv.gz .
Methodology
We will be iterating on the Stanford paper’s implementation, using a Transformer model instead of LSTM. Our hyperparameters will be as such:
Adam optimizer with learning rate of 0.01. Batch size of 32. (same as paper)
Dropout of 0.1 (but will be experimented with).
Hidden layer sizes of 300 units.

Our architecture is as such:
An embedding layer.
A transformer OR bidirectional LSTM layer.
A dense Sigmoid layer.

Metrics
What constitutes success?
We consider success to be a model that will train on our data set and achieve ~70 accuracy on sentiment analysis
Trend: accuracy is going up (shows model is learning)
What experiments do you plan to run?
Since we don’t have the time or computational resources available to us that the original researchers did, we plan on experimenting with different sized units, training size samples, as well as architecture.
Attention is All You Need released two months after this paper, so it may be possible to rework the model using transformers
For assignments, we have looked at the accuracy of the model. Does the notion of accuracy apply for your project?
Yes, we can check if the sentiment analysis of a review matches with whether the review is “positive” or “negative”
If you are implementing an existing project, detail what the authors of that paper were hoping to find and how they quantified the results of their model.
The authors were trying to develop an unsupervised NN that can detect sentiment in text, and used Amazon review ratings to analyze their accuracy

## Results

Transformer model: 77.6% accuracy.
LSTM model: 76.8% accuracy.
