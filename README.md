# Course Recommender System IBM

## Overview of project

The aim of this project was to use various machine learning models for a course recommender system. The dataset which was used was the IBM Course Recommendations dataset which is a collection of data related to IBM courses (specifically those on python) and the interactions students had with them (these were saved as CSVs).

For this project both supervised and unsupervised models were used which were then compared with each other. The project itself consists of several notebooks (listed below) . This project was part of the work done for the IBM Machine Learning Professional certificate. An ERD is also included

The machine learning models used for this project included:

##### Unsuperivsed Learning

- ***Using dot product to compare vectors for recommendations***
- ***Using Bag of Words (Bows) and a similarity matrix***
- ***Clustering and PCA***

#### Supervised Learning


- ***KNN from surprise library***
- ***NMF from surprise library***
- ***Tensor flow Neural Network classifier using embeddings***


# list of Notebooks in this project:

- EDA IBM capstone project
- Unsupervised (course similarity) IBM Capstone project
- Unsupervised (course vectors) IBM capstone project
- Unsupervised (clustering) IBM capstone project
- Supervised (KNN) IBM capstone project
- Supervised (NMF) IBM capstone project
- Supervised (Neural Network) IBM capstone project



# Flowcharts of notebooks in project


### Unsupervised Learning
- Unsupervised (clustering) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_062328](https://user-images.githubusercontent.com/68299933/215196386-54d62240-94dc-46f8-bd72-3c43a53ea530.png)

- Unsupervised (course similarity) IBM Capstone project

Flowchart of notebook:
![Screenshot_20230106_113111](https://user-images.githubusercontent.com/68299933/215196627-b730303a-4079-4c8c-814f-05517a1a1ce7.png)

- Unsupervised (course vectors) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_114034](https://user-images.githubusercontent.com/68299933/215196933-e54e5f8e-7748-49ea-a598-d1778b4c32f8.png)



### Supervised Learning
- Supervised (KNN) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_114406](https://user-images.githubusercontent.com/68299933/215197123-0f4d8421-f24a-4a90-8937-1c599199f24c.png)

- Supervsied (NMF) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_121950](https://user-images.githubusercontent.com/68299933/215197225-c9149ca0-fdfe-4b46-889f-93d1a108bf70.png)

- Supervised (Neural Network) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_122119](https://user-images.githubusercontent.com/68299933/215197276-298b1169-f496-46f2-8326-bf0df7c9b4ae.png)




# Summary of each notebook

## "EDA IBM capstone project" Notebook

#### The libraries used for this notebook were:

- ***Pandas***
- ***numpy***
- ***matplotlib***
- ***seaborn***
- ***wordcloud*** which was used to import ***WordCloud, STOPWORDS and ImageColorGenerator***

#### Summary and visualisations
This notebook conducted an Exploratory Data Analysis (EDA) of the data. This was done through:

- A bar chart to obtain the balance of genres in the dataset (figure 1)

![Screenshot_20230106_063319](https://user-images.githubusercontent.com/68299933/215202863-acab82ed-4269-4560-b3d8-a89d097f7a07.png)
(figure 1)

- A histogram plot to check the distribution of the dataset (figure 2)

![Screenshot_20230106_063601](https://user-images.githubusercontent.com/68299933/215203057-7f5b77d4-8993-4d7b-87f2-02e7a7c88664.png)
(figure 2)

-A dataframe of the 20 most popular courses to see what courses are most likely to be recommended to users (figure 3)

![Screenshot_20230106_094652](https://user-images.githubusercontent.com/68299933/215203436-13c22dc6-e48a-4d84-bc07-6ec94d85bd42.png)
(figure 3)

-A word cloud (figure 4) to visually see what key words appear the most and used stop words was used to eliminate common English words
![Screenshot_20230106_082341](https://user-images.githubusercontent.com/68299933/215203717-6c720539-ef07-4165-874e-419b803afe49.png)
(figure 4)

It should be noted that a Heat map (figure 5) was generated in the "Unsupervised (clustering) IBM capstone project" notebook to see how closely linked the courses were with their content
![Screenshot_20230106_165537](https://user-images.githubusercontent.com/68299933/215204865-a680ff3c-124c-4f76-9c45-a6f5eeca2595.png)


--------------------

## "Unsupervised (course similarity) IBM Capstone project" Notebook

***Note: The code in this notebook needs to be refactored to improve readability and performance***

#### The libraries used for this notebook were:


- ***Pandas***
- ***numpy***
- ***matplotlib***
- ***seaborn***
- ***nltk*** which was used to import ***word_tokenize and stopwords***
- ***gensim*** which was used to import ***corpora and Dictionary***

#### Summary
This notebook used similarity scores to recommend courses to users. This is implemented through a series of functions and a similarity matrix (one was given in the dataset although another one was also generated). The average number of course recommendations per user was found and the top 10 courses recommended were also obtained. The hyper parameters of the ML model were tweaked through changes to the score threshold (similarity score of a course) which resulted in 3 different ML models which were compared using their silhouette score. Furthermore a similarity matrix

#### Functions used:
- The "generate_recommendations_for_all" function was used to obtain reccomendations for all users
- The "generate_recommendations_for_one_user" function was used to obtain the reccomendations for a single user and was used in the "generate_recommendations_for_all" function
- The "compute_silhouette_score" function was used to find the silhoette score of the ML models.
- The "get_most_common_item_and_score" function was used to get the most reccomended courses and the average score


--------------------

## "Unsupervised (course vectors) IBM capstone project" Notebook

#### The libraries used for this notebook were:


- ***Pandas***
- ***numpy***
- ***matplotlib***
- ***seaborn***
- ***Sklearn*** which was used to import ***preprocessing***
- ***scipy*** which was used to import ***directed_hausdorff, pdist***

#### Summary
This notebook used user profile and course genres vectors to recommend courses to users. This is done by finding the dot product between the user profile vector and the course genre vector to get a score. 3 Models were generated using score threshold as the hyper parameter and were compared using their silhouette score.

#### Functions used:
- The "generate_recommendation_scores_dot_product" function was used to obtain reccomendations for all users
- The "compute_silhouette_score" function was used to find the silhouette score of the ML models.


--------------------

## "Unsupervised (clustering) IBM capstone project" Notebook

***Note: The code in this notebook needs to be refactored as it was modified to eventually become part of a new project that is being worked on for an automated clustering algorithm**

#### The libraries used for this notebook were:

- ***Pandas***
- ***numpy***
- ***matplotlib***
- ***seaborn***
- ***sklearn*** which was used to import ***Kmeans,StandardScaler,MinMaxScaler, Normalizer, GaussianMixture, GridSearchCV, PCA and silhoette_score***


#### Summary
This notebook used clustering algorithms to recommend courses to users. Three models were created for this particular project although the groundwork has been laid to create many more models. The models were compared using their F1 score and the top 10 recommendations and average number of recommendations per user was also found. A heat map was also generated which is shown in figure 5 in the ""EDA IBM capstone project" Notebook" section.

#### Functions used:
- The "cluster_df_algorithm" function was used to generate clustering models for recommendations for users. The inputs for the function are the "scaler" which selects which scaling model to use(StandardScaler,MinMaxScaler, Normalizer), "cluster_optimizer" which selects which optimizer to use (gridsearch, lowest sum of squares, gap_statistic), and whether PCA should be used.
- The "cluster_item_enrol" function obtains the cluster for each item and the labels for each user
- The "reccomend_unseen" function returns a dictionary with recommendations for unseen courses to users

----------------------------------------

## "Supervised (KNN) IBM capstone project" Notebook

***Note: The code in this notebook needs to be refactored as it was modified to eventually become part of a new project that is being worked on for an automated clustering algorithm**

#### The libraries used for this notebook were:

- ***Pandas***
- ***numpy***
- ***matplotlib***
- ***seaborn***
- ***surprise*** which was used to import ***train_test_split, GridSearchCV, f1_score, Dataset, Reader, accuracy, KNNBasic, defaultdict and KFold***


#### Summary
This notebook used KNN to recommend courses to users. The KNN models used: "msd" similarity and a k of 10, "cosine" similarity and a k of 10 and "cosine" similarity and a k of 20, effectively producing 3 different ML models. The models were compared using their F1 score.

#### Functions used:
- The "precision_recall_at_k" function was used to obtain the precision and recall values of each model

---------------------------------------------

## "Supervsied (NMF) IBM capstone project" Notebook

#### The libraries used for this notebook were:


- ***Pandas***
- ***numpy***
- ***matplotlib***
- ***seaborn***
- ***Surprise*** which was used to import ***NMF, Dataset, Reader, train_test_split and accuracy***
- ***collections*** which was used to import ***defaultdict***

#### Summary
This notebook used NMF to reccomend courses to users. Three models were generated by changing the hyperparemeters:
- n_factors = 15 and n_epochs = 50
- n_factors = 30 and n_epochs = 100
- n_factors = 60 and n_epochs = 200
The models were compared using their F1 score

#### Functions used:
- The "precision_recall_at_k" function was used to Obtain the precision and recall values of each model
- The "average_dicts" function was used to average precision and recalls of a model which was used for the "precision_recall_at_k" function.


---------------------------------------
## "Supervised (Neural Network) IBM capstone project" Notebook

#### The libraries used for this notebook were:


- ***Pandas***
- ***numpy***
- ***matplotlib***
- ***seaborn***
- ***sklearn*** which was used to import ***GridSearchCV, LabelEncoder, f1_score and train_test_split***
- ***tensorflow*** which was used to import ***keras, KerasClassifier, Sequential and Dense***

#### Summary
This notebook used Neural network embeddings to recommend courses to a user. A merged dataframe is created which is used as the basis for the machine learning models, the features are defined and label encoding is used to convert categorical variables into continuous variables. Three models were created via changing their hyper parameters:

- Model 1 had a test size of 0.2 and used the default epochs setting
- Model 2 had a test size of 0.2 and had the epochs set to 10
- Model 3 had a test size of 0.4 and had the epochs set to 20

For each model, a keras classifier was wrapped around the model, GridSearchCV was used to search over the parameter grid and each model was evaluated using the f1_score.

#### Functions used:
- The "build_model" function was used to create the neural network






## Summary of project (Conclusions)
Each model was compared and the results are discussed in the pdf ("Course Recommonder System IBM Presentation as pdf").  The Best model for the supervised model was the Neural network whilst the best unsupervised model was the user profile-based reccomender system. For future work, more tuning on the hyper parameters for the unsupervised model would be
needed such as altering the optimizers and trying other libraries. A Neural Network can also be used for auto encoding the data for KNN clustering which would provide the unsupervised models something to compete against the supervised models.
