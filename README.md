# UFC Predictions
## Background/Goal:
As an avid UFC fan that religiously watches every PPV event and plans to enroll in MMA classes in the future, creating a Data Science project revolving around UFC fighter data was a no brainer. I was inspired by KAGGLE GUY's project when I first initially tried to look up existing datasets on Kaggle. He has a throughly detailed ETL process where he scrapes fight and fighter data from XXX & XXX. His repo is linked here XXX and I was able to refresh the data tothe most current fight by running the command: XXXXX. With this, I did not have to do my own scraping and had a dataset ready to go in several minutes. My goal is to explore this dataset build a binary classification machine learning model that uses fighter data statistics to predict the outcome of fights. The idea is that these stasitics can represent a fighters tedencies, style, strengths and waknesses and can be fed into a model to predict the winner. 

## Data

## Analysis
I first was able to cluster fighters based on their fighting styles by looking at their fighting dimensions of a mix of striking and grappling. I used Sklearn's KMeans Clustering Algorithm and the common "elbow method" to choose the number of k to reach 4 clusters. After analysis of each cluster, I found that there are 4 main fighting styles:
- Fighter's with knock down power
- Fighter's with a grappling heavy approach
- Fighter's that have a volume based striking approach
- Fighter's that are a jack of all trades that use both striking & grappling but a master of none.

For my UFC Predictions analysis, I first created a baseline model to evaluate my machine learning model against.

<b> Baseline Model </b>

I scraped data from https://www.bestfightodds.com to grab betting odds for every UFC fight and for each fighter. My baseline model then chooses the betting favorite as the winning prediction. The accuracy of this appraoch was 63%.

<b> Machine Learning Model </b>

I used Sklearn's train_test_split method to split my data into a training (80%) and test (205) set. I used cross_val_score to evaluate different classification models on the training set and chose the best performing classfier with accuracy as my metric, which resulted in choosing the RandomForestClassifier Model.

I performed feature selection with sklearn's RFECV (Recursive Feature Elmination with Cross Validation) to select the best set of features across the cross validation groups based on the features importance scores. 

I then hypertuned the Random Forest Classifier parameters (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features) with Sklearn's GridSearchCV.

I fitted the final model on the training set and predicted on the testing set and evaluated the model using various classification metrics.

## Results
CONFUSION MATRIX
ROC/AUC CURVE

Our accuracy on our test set is 62%, which performs worse than just going based off the odds that betting lines creates. The difference between our recall and specificity is outstanding, as our recall is low at 25% and our specificity score is high at 90%. This is intuitively correct as our model is having a hard time to predict when the underdog wins. The inherent randomness of fights where every fighter has a punchers chance, and being a sport where anything can happend and anyone can win on a given day, the model is doing a poor job of predicting when the underdog wins.

## Future Work

