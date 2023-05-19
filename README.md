# UFC Predictions
![ufc](https://github.com/ctran2320/UFC_predictions/assets/133697095/c02c49ed-e888-4c04-9842-d1933378f1cb)

## Background/Goal:
As an avid UFC fan that religiously watches every PPV event and plans to enroll in MMA classes in the future, creating a Data Science project revolving around UFC fighter data was a no brainer. I was inspired by [Rajeev Warrier's](https://www.kaggle.com/datasets/rajeevw/ufcdata) project when I first initially tried to look up existing datasets on Kaggle. He has a throughly detailed ETL process where he scrapes fight and fighter data from http://ufcstats.com/statistics/events/completed?page=all. His repo is linked [here](https://github.com/WarrierRajeev/UFC-Predictions) and I was able to refresh the data tothe most current fight by running the command: python -m src.create_ufc_data. With this, I did not have to do my own scraping and had a dataset ready to go in several minutes. My goal is to explore this dataset build a binary classification machine learning model that uses fighter data statistics to predict the outcome of fights. The idea is that these stasitics can represent a fighters tedencies, style, strengths and waknesses and can be fed into a model to predict the winner. 

## Data
<b> Column definitions </b>:
- <mark> R_ and B_ </mark> prefix signifies red and blue corner fighter stats respectively
- _opp_ containing columns is the average of damage done by the opponent on the fighter
KD is number of knockdowns
SIG_STR is no. of significant strikes 'landed of attempted'
SIG_STR_pct is significant strikes percentage
TOTAL_STR is total strikes 'landed of attempted'
TD is no. of takedowns
TD_pct is takedown percentages
SUB_ATT is no. of submission attempts
PASS is no. times the guard was passed?
REV are the number of reversals
HEAD is no. of significant strinks to the head 'landed of attempted'
BODY is no. of significant strikes to the body 'landed of attempted'
CLINCH is no. of significant strikes in the clinch 'landed of attempted'
GROUND is no. of significant strikes on the ground 'landed of attempted'
win_by is method of win
last_round is last round of the fight (ex. if it was a KO in 1st, then this will be 1)
last_round_time is when the fight ended in the last round
Format is the format of the fight (3 rounds, 5 rounds etc.)
Referee is the name of the Ref
date is the date of the fight
location is the location in which the event took place
Fight_type is which weight class and whether it's a title bout or not
Winner is the winner of the fight
Stance is the stance of the fighter (orthodox, southpaw, etc.)
Height_cms is the height in centimeter
Reach_cms is the reach of the fighter (arm span) in centimeter
Weight_lbs is the weight of the fighter in pounds (lbs)
age is the age of the fighter
title_bout Boolean value of whether it is title fight or not
weight_class is which weight class the fight is in (Bantamweight, heavyweight, Women's flyweight, etc.)
no_of_rounds is the number of rounds the fight was scheduled for
current_lose_streak is the count of current concurrent losses of the fighter
current_win_streak is the count of current concurrent wins of the fighter
draw is the number of draws in the fighter's ufc career
wins is the number of wins in the fighter's ufc career
losses is the number of losses in the fighter's ufc career
total_rounds_fought is the average of total rounds fought by the fighter
total_time_fought(seconds) is the count of total time spent fighting in seconds
total_title_bouts is the total number of title bouts taken part in by the fighter
win_by_Decision_Majority is the number of wins by majority judges decision in the fighter's ufc career
win_by_Decision_Split is the number of wins by split judges decision in the fighter's ufc career
win_by_Decision_Unanimous is the number of wins by unanimous judges decision in the fighter's ufc career
win_by_KO/TKO is the number of wins by knockout in the fighter's ufc career
win_by_Submission is the number of wins by submission in the fighter's ufc career
win_by_TKO_Doctor_Stoppage is the number of wins by doctor stoppage in the fighter's ufc career

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

