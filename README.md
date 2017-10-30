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

### CF model experiment
1. top-5 recommendation for tuning the number of neighbor (n)
    * user-based CF
    * item-based CF
    * how to pick the right number? (ROC, accuracy, precision, recall, or all)

### MF model experiment
1. top-5 recommendation for tuning the number of columns (m)
    * predict the rating and recommend based on the ratings
    * how to pick the right number? (ROC, accuracy, precision, recall, RMSE or all)
    * use latent features and fits into regression model for predict ratings (optional)

### Comparison between all models
1. Hypothesis testing for all performance metrics
2. Precision of all models with respect to top-k
    * CF
    * MF 
    * MF + CF
