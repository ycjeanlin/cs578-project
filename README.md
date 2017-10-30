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

### CF model experiment


### MF model experiment


### Comparison between all models
