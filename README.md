
#                                                     Project Name - Apple and Google tweet Sentiment Analysis.

# 1. Business Understanding
## 1.1 Business Overview
In the modern technological era, social media platforms such as Twitter(x) have become powerful sources where users share real-time opinions on brands and products. Companies such as Apple and Google, both global leaders  in technology and innovation, benefit greatly from understanding these public sentiments. Analyzing tweets about them helps reveal consumer opinions, trends and brand perceptions. Since manually reviewing  thousands of tweets is inefficient, automated sentiment analysis provides an effective solution. Classifing tweets as positive,negative or neutral to help companies monitor reputation, improve customer satisfaction  and make informed strategic decisions.

## 1.2 Problem Statement
Twitter(x) is a space where people share their opinions about brands and products. For global technology companies like Apple and Google, these tweets offer valuable insights into customer satisfaction, brand reputation and customer loyalty. However, the large volume of unstructured data makes it difficult to manually analyze the public sentiment in real time. To solve this problem, this project aims at developing an automated sentiment analysis model using Natural Language Processing(NLP) to classify tweets as positive, negative or neutral. This will help the companies better understand consumer perception, respond to feedback quickly and generally improve their products and overall Brand Image.
## 1.3 Business Objectives 
 ### 1.3.1 Main Objective
To build a model that can rate the sentiment of a Tweet based on its content
 ### 1.3.2 Specific Objectives
* To establish patterns and relationships between tweet content and corresponding sentiment categories.
* To identify whether the special characters potray meaningful info.
* To determine the main sentiment drivers.
* To identify the machine learning model that performs best in classifying tweet sentiment by comparing models based on key performance metrics to generate meaningful insights that reflect customer attitude and brand perception in real time.
* To determine which words, phrases or subjects have the greatest influence on whether people see a brand favourably or unfavourably.


 ### 1.3.3 Research Questions
1. What patterns and relationships exist between tweet content and the sentiment categories?
2. Do special characters such as @, # and links carry any meaningful information that affects tweet sentiment?
3. What specific features are the main targets of users' emotions towards apple and google?
4. Which machine learning model performs the best in classifying tweet sentiment based on metrics such as accuracy, F1-score, precision and recall?
5. What are the main words, phrases or themes that drive positive/negative sentiment towards these brands and how do these patterns change over time?

**Stakeholder Audience:** The primary stakeholder for this project was Apple and Google.

## 1.4 Success Criteria
* The project will be successful if it develops an accurate and reliable sentiment classification model that achieves an F1-weighted of 75% and above and maintains balanced precision and recall across all the sentiment classes.
* Success will also be measured by the model's ability to generalize well to unseen data, minimize missclassification between positive and negative tweets and provide actionable insights that help improve customer services and management of the brand.

  ## Data Understanding & Analysis  

### Dataset Overview   

The Judge Emotion About Brands and Products from CrowdFlower, which is hosted on data.world, was used for this project.  This dataset was appropriate since it includes thousands of Twitter postings that reference a variety of items and companies, all of which are labeled with the emotional tone—whether neutral, negative, or positive.  The dataset is perfect for training and assessing natural language processing models because it contains text-based variables including brand name, sentiment label, and tweet content.

Tweets about Apple and Google in particular were taken out for this study in order to examine and contrast how the public feels about these two well-known tech firms.  This goal is supported by the dataset's structure, which offers user-generated, real-world data that records unscripted customer opinions.  Because of this, it provides a strong foundation for investigating consumer attitudes, brand perception, and emotional trends on social media platforms.

| Metric              | Value |
|----------------------|-------|
| Total Records        | 9093 |
| Features             | 3    |
| Target Variable      | `sentiment` (0 = Positive emotion, 1 = Negative emotion, 3 = Neutral) |
| Positive emotion     | 2,978 (32.8%) |
| Negative emotion     | 570 (6.3%) |
| Neutral              |5544 (61.0%) |

---
### Data Cleaning and Analysis Performed  
- Checked for missing values (1 found in `tweet_text`  and 5802 in `emotion_in_tweet_is_directed_at`  column).
- The missing value in tweet_text was dropped and the 5802 were imputed with `Unknown`.
- There were 22 duplicates in the dataset which were dropped. 
- Removed irrelevant words (e.g. links, hashtags).  
- Renamed `is_there_an_emotion_directed_at_a_brand_or_product` to `sentiment`.  
- Most common words; *SXSW, links, rt* were dropped since they were contextual keywords with no meaning to our target variable.
- **Kmeans Clustering** with PCA for dimensionality reduction was used to represents distinct cluster of tweets with similar textual features.  
- Verified class balance and applied `class_weight="balanced", SMOTE and Random Oversampling`  in modeling.  

## Important Visualizations

<img width="824" height="477" alt="image" src="https://github.com/franklinkuira/apple-and-google-sentimentanalysis/blob/main/visuals/common%20words.png" />

<img width="846" height="480" alt="image" src="https://github.com/franklinkuira/apple-and-google-sentimentanalysis/blob/main/visuals/final_confusion_matrix.png" />

<img width="858" height="404" alt="image" src="https://github.com/franklinkuira/apple-and-google-sentimentanalysis/blob/main/visuals/kmeans.png" />

<img width="465" height="382" alt="image" src="https://github.com/franklinkuira/apple-and-google-sentimentanalysis/blob/main/visuals/wordcloud.png" />
---

### Modelling

This project's main modeling strategy for multi-class sentiment categorization was a TF-IDF (Term Frequency–Inverse Document Frequency) and Logistic Regression pipeline.  This approach was selected because it effectively captures word importance in tweets while maintaining computational lightness, and it offers a robust and interpretable baseline for text classification tasks.

 First, TF-IDF was used to convert tweets into numerical feature vectors, which measure the relative importance of particular words in relation to the overall dataset.  After being trained to predict one of three sentiment categories—positive, negative, or neutral—these features were then fed into a logistic regression classifier.  Metrics like accuracy, precision, recall, and weighted F1-score to account for class imbalance were used to assess the model's performance.

 Although the TF-IDF + Logistic Regression model did not meet the target weighted F1-score threshold of 0.75, it showed satisfactory baseline performance.  Class inequality was one major issue, since neutral tweets greatly outweighed either positive or negative ones.  This imbalance probably affected the classifier's ability to correctly predict minority emotion classifications, even when oversampling approaches were used.

 Furthermore, contextual meaning and semantic links between words are not captured by the TF-IDF representation, despite its effectiveness in recognizing often co-occurring keywords.  This restriction led to minor overfitting, which was seen in the about 18% accuracy difference between testing and training results.

 Considering these limitations, the model provides a strong basis for further development.  Hyperparameter tuning, the incorporation of sophisticated language embeddings like Word2Vec, GloVe, or BERT, and the application of data augmentation to improve class balance are examples of possible improvements.  The sentiment analysis model may develop into a more reliable and broadly applicable instrument for analyzing the emotional tone of social media content with these improvements.
 
The following models were implemented and compared.
1. Logistic Regression
2. Random Forest Classifier
3. Multinomial Naive Boyes
4. XGBoost Classifier
5. CNN-LSTM Model
All models were evaluated using cross validation and tested on unseen data.

### Evaluation and Results

| Model | Weighted Precision | Weighted Recall | Weighted F1-score | Accuracy | Train-Test Δ |
|-------|-------------------|-----------------|------------------|----------|--------------|
| Randomized Logistic Regression | 0.676 | 0.655 | 0.662 | 0.655 | — |
| Randomized Logistic Regression (GridSearch) | **0.681** | 0.651 | **0.662** | 0.651 | **0.165** |
| Random Forest | 0.659 | 0.670 | 0.647 | 0.670 | **0.288** |
| Multinomial Naive Bayes | 0.675 | 0.663 | 0.592 | 0.663 | **0.142** |
| XGBoost Classifier | 0.642 | 0.654 | 0.630 | 0.654 | **0.161** |
| CNN-LSTM | 0.627 | 0.644 | 0.629 | 0.644 | **0.264** |

### Summary of findings

- The TF-IDF + Logistic Regression pipeline demonstrates an acceptable `baseline performance in multi-class sentiment classification`. It did not achieve our success criteria of a >75 weighted F1 score.

- While the models developed in this project performed fairly, certain limitations should be noted. The dataset exhibited `class imbalance` that, despite oversampling, may have influenced performance on minority sentiment classes.

- TF-IDF features limited the model’s ability to capture deep contextual meaning in text, and the final model showed `mild overfitting` (~18% accuracy gap between train and test). Additionally, results are based on a single data source, which `may not generalize` across broader domains or linguistic variations.

- The model provides a strong foundation for improvement through `hyperparameter tuning`, `richer embeddings`, and `more balanced data`. With further refinement, it can serve as a reliable tool for analyzing and interpreting emotional tone in text data.

### Final Insights and Recommendations

1. Improve Data Balance and Context Understanding to Enhance Predictive Accuracy EDA revealed class imbalance, with the Neutral class dominating. Models struggled to learn strong signals for Positive and Negative sentiments. Even high-performing models showed reduced recall for minority classes, meaning customer dissatisfaction might be underrepresented. CNN-LSTM and XGBoost underperformed due to limited semantic context in textual representations (TF-IDF)
2. Use Model Insights to Inform Strategic Business Decisions Sentiment trends extracted by the model can be mapped to specific product features, services, or campaigns identified during text cleaning and feature extraction. Positive emotions indicate brand loyalty drivers; negative emotions pinpoint service gaps.
3. Maintain Continuous Model Evaluation and Retraining for Long-Term Business Value Customer language evolves (slang, emojis, abbreviations), and static models degrade over time. Periodic re-evaluation maintains model relevance and ensures that accuracy remains above the success threshold. this ensures automated sentiment analytics can be trusted and also continued alignment between customer voice and business strategy


### Conclusion 

By implementing a variety of machine learning and deep learning models to categorize tweets into positive, negative, and neutral groups, this study aimed to examine public opinion toward Apple and Google using data from Twitter.  The objective was to assess the efficacy of several sentiment categorization techniques and learn how consumers express ideas about two of the most significant technology businesses.

Model performance was impacted by the imbalanced dataset, which included a large proportion of neutral tweets, according to the analysis.  With a weighted F1-score of 0.662, the TF-IDF + Logistic Regression (Grid Search optimized) pipeline produced the most reliable and comprehensible results out of all the models that were tested.  Despite being investigated, more sophisticated models like CNN-LSTM, XGBoost, and Random Forest did not much outperform the optimized logistic regression model.  This implies that conventional machine learning techniques can compete on sentiment datasets of a moderate size if they are appropriately adjusted.

All things considered, this study shows how machine learning and natural language processing (NLP) methods can be used to obtain insightful information from social media conversations.  It offers a valuable starting point for comprehending how consumers see well-known IT companies and for creating more powerful tools for sentiment and opinion analysis in the digital age.
