#!/bin/bash

mkdir data
aws s3 cp s3://amazon-reviews-pds/tsv/sample_us.tsv ./data
aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Camera_v1_00.tsv.gz ./data
gunzip ./data/amazon_reviews_us_Camera_v1_00.tsv.gz
mv ./data/amazon_reviews_us_Camera_v1_00.tsv ./data/amazon_camera_reviews.tsv