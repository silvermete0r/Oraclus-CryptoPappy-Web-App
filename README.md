# Astana IT University: Blockchain Hack 2023 

## Project name

Oraclus CryptoPappy WebApp

## Selected problem

* **Crypto Market News Impact Analyzer: Correlation of News Sentiment with the Movement of Token Prices**

* **Task:** To develop a tool that effectively correlates the mood of cryptocurrency-related news with the corresponding changes in token prices, eliminating the current discrepancy between the impact of news and market trends.

## Team name

**AIturbo**

## Participants

* Full name: Arman Zhalgasbayev. Email: 220650@astanait.edu.kz
* Full name: Tolegen Aiteni. Email: 221640@astanait.edu.kz
* Full name: Kalkabay Yerkhat. Email: 220304@astanait.edu.kz

## Abstract

**CryptoPappy** is an interactive application that provides detailed market analysis of cryptocurrencies in the form of charts, visualizations, metrics, tables, etc. The main feature of the program is to receive current news on bitcoin in real time. Depending on the quality and quantity of news, the platform provides a forecast of the bitcoin price for the next day.

**The ML model was trained using Tensorflow on architecture:**
``` Python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
```

**My Kaggle Notebook:** https://www.kaggle.com/code/armanzhalgasbayev/eda-how-crypto-news-affects-on-cryptocurrencies

## Demo video

[Link to a demo video showcasing your project, if any. Ensure it is less than 3 minutes long.]

## How to run

### Prerequisites:

 - pandas<=2.0.2
 - Requests<=2.31.0
 - streamlit<=1.23.1
 - tensorflow<=2.15.0.post1
 - transformers<=4.36.2
 - yfinance<=0.2.33


### Running

[Provide specific commands and environment for building and running your project, preferably in a containerized environment.]

Basic example:
```bash
# Clone the repository
git clone [repository-link]

# Navigate to the project directory
cd [project-directory]

# Install Prerequisites
pip install -r requirements.txt

# Run Streamlit App
streamlit run app.py
```

## Inspirations

Since the beginning of 2023, I am engaged in Data Science and Machine Learning. I really enjoy doing this and I improve my skills on the Kaggle platform every day. I had a goal to become an expert at Kaggle, and within a year, I became an expert on Notebooks.

I liked the fact that interesting DS/ML tasks came to the Blockchain Hackathon, it was difficult for me to choose which task to do. But after analyzing all the options, I decided to do a task from Oracle to identify the relationship between the news and the current price of cryptocurrencies.

## Technology stack and organization

* **Programming Language:** `Python`;
* **Frameworks used:** `streamlit`, `tensorflow`;
* **Libraries used:** `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`;

## Solutions and features implemented

While solving this problem several options for the implementation of this project came to my mind at once. But after testing different options, I decided to focus on semantic text analysis using HuggingFace transformers for sentimental analysis. The data collection and processing took a sufficient amount of time. Fortunately, we found a good dataset on the Kaggle platform, in which everything was quite structured. We took historical price data from the official Yahoo Finance resource. 

The logic of the model is simple, we parse all reliable resources by the keyword bitcoin, then check them by sentimental analysis. Depending on the ratio of negative, neutral and positive news, the ML model predicts how much the price of bitcoin will fall or rise.

As a result of all the work, we have trained a fairly good model for predicting the price of bitcoin the next day and we have developed a whole platform for analyzing cryptocurrencies and news, as well as for predicting the price of bitcoin the next day.

* Full Kaggle Notebook for Tensorflow ML Model Training: https://www.kaggle.com/code/armanzhalgasbayev/eda-how-crypto-news-affects-on-cryptocurrencies 

* To get the latest news, we developed our own parser based on the News API separately: https://github.com/silvermete0r/Crypto-News-Parsing-Guidline

## Challenges faced

The main work was to develop a machine learning model to predict the price of bitcoin depending on the news. In the process, there were problems with analyzing this data, how to train the model better in order to improve the quality of the model. I tested different options, it turned out to be the best to conduct a semantic analysis of the news around the world for each day and make an analysis of the ratio of the number of positive, negative and neutral news per day. We tried to do Named Entity Recognition, but it didn't help, but on the contrary, it worsened the quality.

## Lessons learned

* Data Science / Machine Learning Practice (Tensorflow/Keras, NLP Techniques);
* Hackathon practice, gaining new experience and knowledge;
* Increasing competitiveness;

## Future work

* To work on improving the model, it is definitely possible to make an even better model based on `word2seq` & The `seq2seq` architecture.

## Additional sources

**Yahoo Finance:** [Top CryptoCurrencies](https://finance.yahoo.com/crypto?guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAFdVCpPk-mtL4GaFSZ_fyIGd1q9bNPXRVtgQdYERukDz6_ZPbubXewi7H442lbIXpiiWUPlxIEpHyWyHSn84oTa2FyOagzbsiigb7MMcb-2VarhtPWcrqA4YKC5WICbyHpU66DbREH_7Li2fE9RcyUgLVfiTVYsVHRa_c5UgPPut) 
**Yahoo | Kaggle Dataset for Bitcoin:** https://www.kaggle.com/datasets/armanzhalgasbayev/bitcoin-historical-data-2021-2023
**Cypto News+ Dataset:** https://www.kaggle.com/datasets/oliviervha/crypto-news
**Streamlit Docs:** https://docs.streamlit.io/
**Tensorflow Docs:** https://www.tensorflow.org/api_docs
