# UFC Predictions
![ufc](https://github.com/ctran2320/UFC_predictions/assets/133697095/c02c49ed-e888-4c04-9842-d1933378f1cb)

## Background/Goal:
As an avid UFC fan that religiously watches every PPV event and plans to enroll in MMA classes in the future, creating a Data Science project revolving around UFC fighter data was a no brainer. I was inspired by [Rajeev Warrier's](https://www.kaggle.com/datasets/rajeevw/ufcdata) project when I first initially tried to look up existing datasets on Kaggle. He has a throughly detailed ETL process where he scrapes fight and fighter data from http://ufcstats.com/statistics/events/completed?page=all. His repo is linked [here](https://github.com/WarrierRajeev/UFC-Predictions) and I was able to refresh the data tothe most current fight by running the command: python -m src.create_ufc_data. With this, I did not have to do my own scraping and had a dataset ready to go in several minutes. My goal is to explore this dataset build a binary classification machine learning model that uses fighter data statistics to predict the outcome of fights. The idea is that these stasitics can represent a fighters tedencies, style, strengths and waknesses and can be fed into a model to predict the winner. 

## Data
<b> Column definitions </b>:
- <b> R_ and B_ </b> prefix signifies red and blue corner fighter stats respectively
- <b> _opp_ </b> containing columns is the average of damage done by the opponent on the fighter
- <b> KD </b> is number of knockdowns
- <b> SIG_STR </b> is no. of significant strikes 'landed of attempted'
- <b> SIG_STR_pct </b> is significant strikes percentage
- <b> TOTAL_STR </b> is total strikes 'landed of attempted'
- <b> TD </b> is no. of takedowns
- <b> TD_pct </b> is takedown percentages
- <b> SUB_ATT </b> is no. of submission attempts
- <b> PASS </b> is no. times the guard was passed?
- <b> REV </b> are the number of reversals
- <b> HEAD </b> is no. of significant strinks to the head 'landed of attempted'
- <b> BODY </b> is no. of significant strikes to the body 'landed of attempted'
- <b> CLINCH </b> is no. of significant strikes in the clinch 'landed of attempted'
- <b> GROUND </b> is no. of significant strikes on the ground 'landed of attempted'
- <b> win_by </b> is method of win
- <b> last_round </b> is last round of the fight (ex. if it was a KO in 1st, then this will be 1)
- <b> last_round_time </b> is when the fight ended in the last round
- <b> Format </b> is the format of the fight (3 rounds, 5 rounds etc.)
- <b> Referee </b> is the name of the Ref
- <b> date </b> is the date of the fight
- <b> location </b> is the location in which the event took place
- <b> Fight_type </b> is which weight class and whether it's a title bout or not
- <b> Winner </b> is the winner of the fight
- <b> Stance </b> is the stance of the fighter (orthodox, southpaw, etc.)
- <b> Height_cms </b> is the height in centimeter
- <b> Reach_cms </b> is the reach of the fighter (arm span) in centimeter
- <b> Weight_lbs </b> is the weight of the fighter in pounds (lbs)
- <b> age </b> is the age of the fighter
- <b> title_bout </b> Boolean value of whether it is title fight or not
- <b> weight_class </b> is which weight class the fight is in (Bantamweight, heavyweight, Women's flyweight, etc.)
- <b> no_of_rounds </b> is the number of rounds the fight was scheduled for
- <b> current_lose_streak </b> is the count of current concurrent losses of the fighter
- <b> current_win_streak </b> is the count of current concurrent wins of the fighter
- <b> draw </b> is the number of draws in the fighter's ufc career
- <b> wins </b> is the number of wins in the fighter's ufc career
- <b> losses </b> is the number of losses in the fighter's ufc career
- <b> total_rounds_fought </b> is the average of total rounds fought by the fighter
- <b> total_time_fought(seconds) </b> is the count of total time spent fighting in seconds
- <b> total_title_bouts </b> is the total number of title bouts taken part in by the fighter
- <b> win_by_Decision_Majority </b> is the number of wins by majority judges decision in the fighter's ufc career
- <b> win_by_Decision_Split </b> is the number of wins by split judges decision in the fighter's ufc career
- <b> win_by_Decision_Unanimous </b> is the number of wins by unanimous judges decision in the fighter's ufc career
- <b> win_by_KO/TKO </b> is the number of wins by knockout in the fighter's ufc career
- <b> win_by_Submission </b> is the number of wins by submission in the fighter's ufc career
- <b> win_by_TKO_Doctor_Stoppage </b> is the number of wins by doctor stoppage in the fighter's ufc career

## Analysis
I first was able to cluster fighters based on their fighting styles by looking at their fighting dimensions of a mix of striking and grappling. I used Sklearn's KMeans Clustering Algorithm and the common "elbow method" to choose the number of k to reach 4 clusters. After analysis of each cluster, I found that there are 4 main fighting styles:
- Fighter's with knock down power
- Fighter's with a grappling heavy approach
- Fighter's that have a volume based striking approach
- Fighter's that are a jack of all trades that use both striking & grappling but a master of none.

For my UFC Predictions analysis, I first created a baseline model to evaluate my machine learning model against.

<b> Baseline Model </b>

I scraped data from https://www.bestfightodds.com to grab betting odds for every UFC fight and for each fighter. My baseline model then chooses the betting favorite as the winning prediction. The accuracy of this appraoch was 63.2%.

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

