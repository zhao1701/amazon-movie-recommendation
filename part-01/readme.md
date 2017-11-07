# A Comparison of Recommendation Models for Amazon's Movie and TV Catalog
#### by Charissa Ding, Carrie Yang, and Derek Zhao

The use of personalization has grown significantly to enhance the customer experience in various industries and has become especially indispensable in e-commerce. An online retailer that is able to make high quality recommendations for its customers is likely to sell more products and thus increase revenue and, by extension, profit. And if said retailer earns a reputation for providing accurate recommendations, it will be more effective at retaining customers and fostering trust about future recommendations, further increasing sales, revenue, and profit in the long run.

In this project, we survey the effectiveness of various traditional recommendation methods by testing their performance on varying-sized samples of the Amazon Movie and TV Reviews dataset.

#### Contents

- [Data](#data)
- [Evaluation Metrics](#evaluation-metrics)
- [Models](#models)
- [Methodology](#methodology)
- [Results](#results)
- [Concluding Remarks](#concluding-remarks)

## Data

#### Source and structure
The data used for this project comes from a collection of [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/) maintained by Dr. Julian McAuley and the University of California at San Diego. The Movie and TV dataset from this collection contains reviews of films and TV series available for purchase in Amazon's catalog. Each review was written between **May 1996** and **July 2014** and contains the following relevent features:

- **reviewerID**: An alphanumeric string that uniquely identifies the user/author of the review.
- **asin**: The Amazon Standard Identification Number is a 10-character alphanumeric string that uniquely identifies a product within Amazon's catalog
- **rating**: An integer from 1 to 5, inclusive, that indicates the number of stars a user has rated for a particular item, with larger numbers corresponding to increased favorability.
Additional features such as the text of the review or the number of helpfulness votes of the review are also included in the dataset, however, traditional recommender system models are not suited for utilizing this data, so it is excluded from this project.

#### Summary
Amazon's Movie and TV Review dataset consists of **1.7 million** explicit ratings provided by **123,960** unique users for **50,052** unique items. The data has been pre-filtered so that all users have given at least 5 ratings and all items have received at least 5 ratings. Despite the pre-filtering, there are only 1.7 million available ratings for **6 billion** user-item pairs, resulting in a low density: **0.027%**. Such a high level of sparsity can be attributed to the fact that the dataset only contains ratings for those users who have written a review for an item and does not contain ratings for those users that only give star ratings. Thus, the density of the dataset is artificially low. In conjunction with the fact that such a low density presents challenges in generating suitable sample sets, the dataset is carefully re-filtered to produce a higher density using a procedure explained in greater detail later.

#### Exploratory Data Analysis
The distribution of ratings in the Amazon Movie and TV dataset is heavily skewed towards 5-star ratings. In fact, over half of the ratings are 5 stars, pulling the average rating up to **4.11** stars. Such an unbalanced distribution may prove challenging when trying to treat the data as a collection of explicit ratings, since the more dominant a single rating value becomes, the more the dataset resembles a collection of binary implicit ratings. Nonetheless, we proceed as planned, keeping in mind that binarizing the ratings remains an option for a future project.

![](imgs/dist-ratings.png)

We also observe the typical long-tail distribution when calculating the number of reviews written by each reviewer: a small number of users are highly prolific while the vast majority of users write only a few reviews. In fact, the average and median number of reviews per user is **14** and **7**, respectively.

<p align='center'>
  <img src='imgs/num-reviews-per-user.png'>
  Note: reviewer rank refers to the rank of the reviewer in terms of the number of reviews that reviewer has written.
</p>

The same pattern can be observed in the number of reviews per item. Because there are far fewer items than reviewers, we naturally expect that the median and average review count per item, **13** and **33.9** respectively, is higher than that per reviewer.

<p align='center'>
  <img src='imgs/num-reviews-per-item.png'>
  Note: item rank refers to the rank of the item in terms of the number of reviews that item has received.
</p>

In addition, we observe that over **90%** of reviewers give average star ratings above a **3**. A typical reviewer rates items with an average of **4.2 stars**. Intuitively, this makes sense because people tend to only purchase items already like or believe they will like. It is also possible that the act of purchasing an item predisposes a customer to liking it.

Finally, it is worth mentioning the number of reviews per month is relatively stable from December 1999 to June 2012. On September 4, 2012, Amazon signed a deal with Epix to feature a much larger library of popular movies that may have contributed to the dramatic rise in monthly review volume.

![](imgs/num-reviews-over-time.png)

## Models
In the context of a recommendation problem, a predictive model is an algorithm capable of using existing user-item ratings, such as those already present in the dataset, to infer missing user-item ratings (i.e. how a user would rate an item they have not actully rated). For the problem of generating recommendations from the Amazon Movie and TV Reviews dataset, we consider 4 baseline models and 3 traditional models:

- **Averaging baseline**: The averaging baseline simply calculates the overall average of all ratings and uses this value as the prediction for all missing ratings. It is the crudest of all models.
- **Manual baseline**: The manual baseline is a more nuanced baseline where a predicted rating is calculated as the sum of the overall average, user bias, and item bias. A user's bias is manually calculated as the average of all ratings that a user has made subtracted from the overall average. Similarly, an item's bias is manually calculated as the average of all ratings that an item has received subtracted from the overall average.
- **Stochastic gradient descent (SGD) baseline**: The [SGD baseline](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly) is one of two *learned* baseline models. It is the same as the manual baseline model, except rather than explicitly calculating values for each user and item bias, the user and item biases are parameters learned through optimizing an objective function using SGD.
- **Alternating least squares (ALS) baseline**: The [ALS baseline](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly) is essentially the SGD baseline but with ALS replacing SGD as the method of optimization.
- **Unconstrained Matrix Factorization (SVD)**: Unconstrained matrix factorization, commonly but incorrectly known as [singular value decomposition](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD), is a method wherein a full ratings matrix is approximated as the product of matrices of latent vectors, which are learned from the available ratings. Matrix factorization techniques are particularly well-suited for inferring missing ratings in a high sparsity ratings matrix.
- **Nonnegative Matrix Factorization (NMF)**: [NMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF) is a variant of SVD with the constraint that all latent vectors may only contain positive values.
- **Item-based K-Neighbest Neighbors with Normalization**: [KNN](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithZScore) is a predictive method where the inferred rating for a user-item pair is calculated as a linear combination of the ratings for the k most similar items that user has already rated. The weights in the linear combination are simply the normalized similarities between the target item and the rated items.

## Evaluation Metrics
Below we define the metrics used to evaluate the performance of each predictive model. In the context of an online retailer making purchase recommendations to its customers, the following four evaluation criterion are particularly relevant:
1. **Mean absolute error (MAE)**: 
