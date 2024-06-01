### Problem Statement

Given a directed social graph, predict missing links to recommend users (Link prediction in the graph).

### Data Overview

The dataset is taken from Facebook's recruiting challenge on Kaggle - https://www.kaggle.com/c/FacebookRecruiting <br>
The dataset contains two columns source and destination eac edge in the graph.
- source_node (int64)
- destination_node (int64)

### Mapping the Problem into a Supervised Learning Problem

Generated training samples of good and bad links from the given directed graph and for each link got some features like no. of followers, are they followed back, page rank, Katz score, Adar index, some svd features of adj matrix, some weight features, etc. and trained ml model based on these features to predict link.

### Business Objectives and Constraints

- No low-latency requirement.
- Probability of prediction is useful to recommend the highest probability links.

### Performance Metrics for Supervised Learning

- Both precision and recall are important so F1-score is a good choice.
- Confusion matrix.
