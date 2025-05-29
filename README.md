## NexusIndex: A Self-Optimizing Multimodal Framework for Fake News Detection  
**Accepted at [IEEE MIPR 2025](https://sites.google.com/view/mipr-2025/calls/call-for-papers?authuser=0)**

## Authors  
- [Solmaz Seyed Monir](https://students.washington.edu/solmazsm/), University of Washington  
- [Dr. Dongfang Zhao](https://hpdic.github.io/), University of Washington  
- [Dr. Yan Bai](https://directory.tacoma.uw.edu/employee/yanb), University of Washington  

---

## Overview

NexusIndex is an advanced fake news detection framework that integrates multimodal embeddings, vectorized proximity layers, and the FAISSNexusIndex layer to significantly enhance retrieval efficiency and detection accuracy. It leverages Transformer-based models for text and MobileNet V3 for image analysis, combined with an adaptive semi-supervised learning approach that dynamically refines the model with evolving misinformation.

## Features

- **Multimodal Embedding**: Utilizes Transformer-based models for text embeddings and MobileNet V3 for image embeddings.
- **FAISSNexusIndex Layer**: Efficient high-dimensional similarity retrieval integrated into the learning process.
- **Adaptive Semi-Supervised Learning**: Incorporates pseudo-labeling and Local Variance Filtering (LVF) to adapt to new misinformation.
- **Real-Time Detection**: Optimized for fast, real-time similarity searches and fake news detection.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/solmazsm/NexusIndex.git
   cd NexusIndex

# Dataset

We evaluate NexusIndex on several datasets, including:

Politifact: A well-known dataset for fact-checking. <a href="https://github.com/solmazsm/FakeNewsNet-data">Politifact</a>

GossipCop: A dataset containing news articles, labeled as real or fake.<a href="https://github.com/solmazsm/FakeNewsNet-data">GossipCop</a>

huggingface: The GossipCop dataset is available on Hugging Face. <a href="https://huggingface.co/datasets/osusume/Gossipcop/viewer/default/train?row=27&views%5B%5D=train">huggingface</a>

ABC News: A large-scale dataset used for semi-supervised learning with pseudo-labeling.
WELFake: A text-based dataset containing real and fake news articles.
The datasets are used to train and test the fake news detection model.


<a href="https://components.one/datasets/all-the-news-2-news-articles-dataset">All the News:</a> This dataset contains 2,688,878 news articles and essays from 27 American publications, spanning January 1,2016 to April 2, 2020. It is an expanded edition of the original All the News dataset, which was compiled in early 2017. While the original dataset contains more than 100,000 articles, the new dataset’s greater size and breadth should allow researchers to study a wider selection of media.
To enhance the performance of fake news detection, we propose integrating a threshold-based pseudolabeling strategy within the NexusIndex framework. This approach begins by training the initial model on a labeled dataset, followed by
applying the trained model to predict probabilities on an unlabeled
dataset.

<a href="https://huggingface.co/datasets/davanstrien/WELFake">WELFake:</a> The WELFake dataset consists of 72,134 news articles, with 35,028 classified as real and 37,106 as fake. This dataset was created by merging four well-known news datasets: Kaggle, McIntire, Reuters, and BuzzFeed Political. The goal of this merger was to mitigate the risk of overfitting in machine learning classifiers and to offer a larger corpus of text data to enhance the training process for fake news detection models.

Dataset contains four columns: Serial number (starting from 0); Title (about the text news heading); Text (about the news content); and Label (0 = fake and 1 = real).

There are 78098 data entries in csv file out of which only 72134 entries are accessed as per the data frame.

# Evaluation Metrics
NexusIndex evaluates the model performance using several metrics:

- **Accuracy**: The percentage of correctly classified news articles.
- **Precision**: The percentage of true positive predictions among all positive predictions.
- **Recall**: The percentage of true positive predictions among all actual positive instances.
- **F1-Score**: The harmonic mean of Precision and Recall.
- **AUC**: Area under the receiver operating characteristic curve.
