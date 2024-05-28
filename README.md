
# NexusIndex #CIKM 2024
# We Propose NexusIndex framework integrates advanced NLP models with a neural network classifier, providing a comprehensive system for fake news detection.
# NexusIndexModel, which you describe as a neural network with three fully connected layers and sigmoid activation

This folder contains all the necessary implementation.
# Dataset
NexusIndex Dataset: The NexusIndex dataset is a collection of news articles that have been indexed using vector embeddings for efficient search and retrieval. It contains a vast array of articles spanning various topics and sources, providing researchers with a rich corpus for exploration and analysis.

The dataset encompasses news articles from multiple sources, including but not limited to those compiled by the WELFake and the All the News dataset. 



# Download Dataset
Instructions on how to download the dataset can be found here.

<a href="https://components.one/datasets/all-the-news-2-news-articles-dataset">All the News:</a> This dataset contains 2,688,878 news articles and essays from 27 American publications, spanning January 1,2016 to April 2, 2020. It is an expanded edition of the original All the News dataset, which was compiled in early 2017. While the original dataset contains more than 100,000 articles, the new dataset’s greater size and breadth should allow researchers to study a wider selection of media.
To enhance the performance of fake news detection, we propose integrating a threshold-based pseudolabeling strategy within the NexusIndex framework. This approach begins by training the initial model on a labeled dataset, followed by
applying the trained model to predict probabilities on an unlabeled
dataset.


<a href="https://huggingface.co/datasets/davanstrien/WELFake">WELFake:</a> The WELFake dataset consists of 72,134 news articles, with 35,028 classified as real and 37,106 as fake. This dataset was created by merging four well-known news datasets: Kaggle, McIntire, Reuters, and BuzzFeed Political. The goal of this merger was to mitigate the risk of overfitting in machine learning classifiers and to offer a larger corpus of text data to enhance the training process for fake news detection models.

Dataset contains four columns: Serial number (starting from 0); Title (about the text news heading); Text (about the news content); and Label (0 = fake and 1 = real).

There are 78098 data entries in csv file out of which only 72134 entries are accessed as per the data frame.

# Evaluation Implementation



For any questions, concerns, or comments for improvements, etc, please create an issue on the issues page for this project, or email the authors directly.
