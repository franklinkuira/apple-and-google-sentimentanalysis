# apple-and-google-sentimentanalysis

# ✨ Overview of the Project 

The goal of this research is to analyze sentiment in tweets about Apple and Google in order to determine how the general public feels about these two significant tech firms.  The analysis uses Natural Language Processing (NLP) approaches to categorize user sentiments as neutral, negative, or positive using data gathered from Twitter.

 This study compares the public's perceptions of Apple and Google based on debates and opinions expressed in real time on social media.  Understanding consumer satisfaction, market sentiment, and brand reputation may all benefit from this knowledge.
# 🧑‍⚕️ Business and Data Understanding 

## Introduction

Social media has developed into a powerful forum for individuals to debate businesses and goods, share their experiences, and voice their opinions.  Because of its vast user base and real-time nature, Twitter stands out among these networks as a crucial resource for gauging public sentiment.  Businesses frequently keep an eye on social media discussions to manage brand reputation, assess public opinion, and guide marketing or strategic choices.

 In this study, we use data gathered from Twitter to examine public opinion toward Apple and Google, two of the top tech businesses in the world.  Both businesses are well-known throughout the world and have a big influence on consumer technology, innovation, and digital ecosystems.  Public perception of their goods, services, and policies, however, can differ greatly and shift quickly over time.
 This project attempts to categorize tweets mentioning Apple and Google into positive, negative, or neutral attitudes using sentiment analysis and Natural Language Processing (NLP) techniques.  Understanding how consumers view these brands, comparing general sentiment trends among them, and investigating possible causes of differences in public opinion are the objectives.

 The investigation's conclusions can be used to show how social media mirrors actual brand perception and highlight the usefulness of data-driven sentiment analysis in figuring out how consumers feel in the digital era.

**Stakeholder Audience:** The primary stakeholder for this project was Apple and Google.

## Problem Statement

* Social media sites like Twitter have become into influential forums in the current digital age where customers may openly express their thoughts about goods, services, and businesses.  Since public opinion has a direct impact on brand reputation, consumer loyalty, and market success, it is imperative for multinational technological behemoths like Apple and Google to comprehend these online opinions.

 However, it is difficult to manually analyze and interpret public opinion due to the large amount and unstructured nature of social media data.  Conventional techniques like surveys and interviews take a lot of time, have a small sample size, and frequently miss sentiment patterns in real time.
  
* This study uses sentiment analysis and Natural Language Processing (NLP) techniques to automatically identify and compare public sentiment toward Apple and Google.  The study looks for trends in user sentiments, attitudes, and opinions expressed online by examining tweets that mention these two businesses.

 In an increasingly digital and socially linked world, brand managers, marketers, and researchers who want to know how customers view key technology firms can benefit greatly from the research findings.

## Dataset Choice 

The Judge Emotion About Brands and Products from CrowdFlower, which is hosted on data.world, was used for this project.  This dataset was appropriate since it includes thousands of Twitter postings that reference a variety of items and companies, all of which are labeled with the emotional tone—whether neutral, negative, or positive.  The dataset is perfect for training and assessing natural language processing models because it contains text-based variables including brand name, sentiment label, and tweet content.

Tweets about Apple and Google in particular were taken out for this study in order to examine and contrast how the public feels about these two well-known tech firms.  This goal is supported by the dataset's structure, which offers user-generated, real-world data that records unscripted customer opinions.  Because of this, it provides a strong foundation for investigating consumer attitudes, brand perception, and emotional trends on social media platforms.

## Important Visualizations

<img width="824" height="477" alt="image" src="https://github.com/user-attachments/assets/b02eb946-26b8-4723-88c8-b118871ac429" />

<img width="846" height="480" alt="image" src="https://github.com/user-attachments/assets/2da0e1d0-8303-40eb-82b9-47ceba2f5914" />

<img width="858" height="404" alt="image" src="https://github.com/user-attachments/assets/ccc72131-48b2-4859-b1f0-4a1a0314d672" />

<img width="465" height="382" alt="image" src="https://github.com/user-attachments/assets/d54710bb-9794-446d-a439-8fd567a4c5e7" />

# 🧹 Data Cleaning 

Duplicate items, inconsistent sentiment labels, and unnecessary columns were found during the dataset's initial review, which may have an impact on the analysis's accuracy.  To be more precise, 22 duplicate rows were eliminated in order to prevent repeated tweets from skewing the sentiment distribution or the model training procedure.  Duplicates had to be removed in order to preserve data integrity because keeping them could have resulted in false frequency counts and distorted comparisons between Apple and Google.

 Additionally, the sentiment categories "I can't tell" and "No emotion toward brand or product" were merged into the single label "Neutral."  In order to ensure a more impartial and understandable sentiment analysis, the sentiment categorization process was consolidated into three distinct categories: Positive, Negative, and Neutral.

# ⚙️ Modeling Approach 

This project's main modeling strategy for multi-class sentiment categorization was a TF-IDF (Term Frequency–Inverse Document Frequency) and Logistic Regression pipeline.  This approach was selected because it effectively captures word importance in tweets while maintaining computational lightness, and it offers a robust and interpretable baseline for text classification tasks.

 First, TF-IDF was used to convert tweets into numerical feature vectors, which measure the relative importance of particular words in relation to the overall dataset.  After being trained to predict one of three sentiment categories—positive, negative, or neutral—these features were then fed into a logistic regression classifier.  Metrics like accuracy, precision, recall, and weighted F1-score to account for class imbalance were used to assess the model's performance.

 Although the TF-IDF + Logistic Regression model did not meet the target weighted F1-score threshold of 0.75, it showed satisfactory baseline performance.  Class inequality was one major issue, since neutral tweets greatly outweighed either positive or negative ones.  This imbalance probably affected the classifier's ability to correctly predict minority emotion classifications, even when oversampling approaches were used.

 Furthermore, contextual meaning and semantic links between words are not captured by the TF-IDF representation, despite its effectiveness in recognizing often co-occurring keywords.  This restriction led to minor overfitting, which was seen in the about 18% accuracy difference between testing and training results.

 Considering these limitations, the model provides a strong basis for further development.  Hyperparameter tuning, the incorporation of sophisticated language embeddings like Word2Vec, GloVe, or BERT, and the application of data augmentation to improve class balance are examples of possible improvements.  The sentiment analysis model may develop into a more reliable and broadly applicable instrument for analyzing the emotional tone of social media content with these improvements.
# 📈 Evaluation 

The evaluation of the model focused on:

With a weighted F1-score of 0.662 and modest overfitting, Logistic Regression (with Grid Search optimization) produced the best balanced and comprehensible results out of all the models.  Despite their restricted size and imbalanced datasets, deep learning and ensemble techniques provided gradual gains in contextual knowledge.

 All things considered, the assessment demonstrates that although baseline models offer valuable insights into sentiment patterns, further advancements like class rebalancing, data augmentation, and contextual embeddings would greatly increase the robustness and generalizability of the models for extensive sentiment analysis.

# 🎯 Conclusion 

By implementing a variety of machine learning and deep learning models to categorize tweets into positive, negative, and neutral groups, this study aimed to examine public opinion toward Apple and Google using data from Twitter.  The objective was to assess the efficacy of several sentiment categorization techniques and learn how consumers express ideas about two of the most significant technology businesses.

Model performance was impacted by the imbalanced dataset, which included a large proportion of neutral tweets, according to the analysis.  With a weighted F1-score of 0.662, the TF-IDF + Logistic Regression (Grid Search optimized) pipeline produced the most reliable and comprehensible results out of all the models that were tested.  Despite being investigated, more sophisticated models like CNN-LSTM, XGBoost, and Random Forest did not much outperform the optimized logistic regression model.  This implies that conventional machine learning techniques can compete on sentiment datasets of a moderate size if they are appropriately adjusted.

All things considered, this study shows how machine learning and natural language processing (NLP) methods can be used to obtain insightful information from social media conversations.  It offers a valuable starting point for comprehending how consumers see well-known IT companies and for creating more powerful tools for sentiment and opinion analysis in the digital age.

