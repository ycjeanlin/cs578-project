# CS578 Project

### Preprocess
1. Split data into training and testing data
    * for each user, we randomly sample 80% of his logs for training and 20% of them for testing
    * 80% training / 20% testing (leave testing data for experiments)
2. Split training data into k-fold for parameter tuning
    * for each user, we use (k-1/k) portion of his logs and validate the result by the rest portion of his logs
3. Remove outliers
    * the user with number of ratings < x
    * the movies with number of ratings < y
  
### Data description
1. Basic statistics
    * Number of users
    * Number of ratings
    * Number of movies
    * so on
2. Feature selection

### Performance Metrics
| user    | top-5 list | items in testing | precision         | recall             | hit |
|---------|------------|------------------|-------------------|--------------------|-----|
| 1       | 1,2,3,4,5  | 1,2,9,10         | 2/5               | 2/4                | 1   |
| 2       | 2,3,5,6,7  | 1,2,3,4,5,6,7    | 5/5               | 5/7                | 1   |
| 3       | 1,2,3,4,5  | 6,7,8,9,10       | 0/5               | 0/5                | 0   |
| 4       | 2,3,4,8,9  | 1,2,6,7          | 1/5               | 1/4                | 1   |
| summary | x          | x                | (2/5+5/5+0+1/5)/4 | (2//4+5/7+0+1/4)/4 | 3/4 |

1. Precision
   * Precision= (Relevant_Items_Recommended in top-k) / (k_Items_Recommended)
   * Example: top-5, precision=3/5 means 3 out of 5 recommended items are in the list of items in the testing data of a user.
   * We user average Precision = (\sum_{u \in U} precision_u)/|U|
2. Recall
   * Recall= (Relevant_Items_Recommended in top-k) / (Relevant_Items)
   * Example: recall = 5/7, there are 7 items in the testing data of a user, and 5 of them are in our recommended list
   * We user average Recall = (\sum_{u \in U} recall_u)/|U|
3. Accuracy
   * Accuracy = (number of hits)/(number of testing users)
   * Example: there are 4 users in our testing dataset. 3 of all top-k lists contains the items they prefered. accuracy = 3/4
### CF model experiment
1. top-5 recommendation for tuning the number of neighbor (n)
    * user-based CF
    * item-based CF
    * how to pick the right number? (**ROC**, accuracy, precision, recall, or all)

### MF model experiment
1. top-5 recommendation for tuning the number of columns (m)
    * predict the rating and recommend based on the ratings
    * how to pick the right number? (**ROC**, accuracy, precision, recall, RMSE or all)
    * use latent features and fits into regression model for predict ratings (optional)

### Comparison between all models
1. Precision of all models with respect to top-k (top-5, top-10, top-15, top-20, top-25, top-30)
    * user CF
    * item CF
    * MF
2. Recall of all models with respect to top-k (top-5, top-10, top-15, top-20, top-25, top-30)
    * user CF
    * item CF
    * MF
3. Accuracy of all models with respect to top-k  (top-5, top-10, top-15, top-20, top-25, top-30)
    * user CF
    * item CF
    * MF
3. AUC table
    * user CF
    * item CF
    * MF
4. Hypothesis testing table
     * user CF
    * item CF
    * MF

