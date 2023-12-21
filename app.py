import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import tensorflow as tf
from datetime import datetime
from transformers import pipeline

st.set_page_config(
    page_title = 'DataFlow App',
    page_icon = '📊',
    layout = 'wide',
    initial_sidebar_state = 'auto',
)

@st.cache_resource()
def get_today_cryptonews():
    keyword = 'bitcoin'
    date = datetime.today().strftime('%Y-%m-%d')
    base = 'https://newsapi.org/v2/everything'
    url = f'{base}?q={keyword}&publishedAt={date}&apiKey={st.secrets["NEWS_API_KEY"]}'
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        data = response.json()['articles']
        return data[:50]
    else:
        return None

@st.cache_resource()
def load_sentiment_model():
    sentiment_model = pipeline('sentiment-analysis')
    return sentiment_model

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('models/crypto_nb.h5')
    return model

@st.cache_resource()
def get_today_crypto_data(symbol='BTC-USD'):
    crypto_data = yf.download(symbol, period='1d', interval='1d')
    crypto_data = crypto_data.reset_index()
    crypto_data.rename(columns={'Datetime': 'Date'}, inplace=True)
    crypto_data['Date'] = pd.to_datetime(crypto_data['Date']).dt.strftime('%Y-%m-%d')
    crypto_data['Change'] = (crypto_data['Close'] / crypto_data['Open']) * 100 - 100
    return crypto_data

@st.cache_resource()
def fetch_crypto_data(symbol, start_date, end_date):
    symbol += '-USD'
    crypto_data = yf.download(symbol, start=start_date, end=end_date)
    crypto_data = crypto_data.reset_index()
    crypto_data['Date'] = pd.to_datetime(crypto_data['Date']).dt.strftime('%Y-%m-%d')
    crypto_data['Change'] = (crypto_data['Close'] / crypto_data['Open']) * 100 - 100
    return crypto_data.sort_values(by='Date')

def get_today_cryptonews_sentiment(news):
    sentiment_model = load_sentiment_model()
    sentiments_count = {'negative': 0, 'neutral': 0, 'positive': 0}
    for article in news:
        sentiment = sentiment_model(article['content'])[0]['label']
        sentiments_count[sentiment.lower()] += 1
    summary_df = pd.DataFrame([sentiments_count], columns=sentiments_count.keys())
    summary_df['date'] = datetime.today().strftime('%Y-%m-%d')
    summary_df = summary_df[['date', 'negative', 'neutral', 'positive']]
    return summary_df

def get_prediction(today_cryptonews_sentiment, close_price):
    model = load_model()
    prediction = model.predict(today_cryptonews_sentiment[['negative', 'neutral', 'positive']])[0][0]
    return round(close_price * (100 + prediction) / 100, 2)

def main():
    st.title('📈 CryptoPappy Analysis App')

    st.markdown('''
            * This app retrieves uptodate cryptocurrency data using Yahoo Finance API and performs exploratory data analysis and prediction of Bitcoin closing price for the next day using trained TensorFlow model.
            * **Libraries used:** `streamlit`, `yfinance`, `pandas`, `tensorflow`, `matplotlib`, `seaborn`
            * **Data source:** [Yahoo Finance](https://finance.yahoo.com/)
    ''')
    st.write('---')

    st.sidebar.image('media/logo.png', use_column_width=True)

    st.sidebar.header('User Input')
    selected_crypto = st.sidebar.selectbox('Select Cryptocurrency', ['BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'XRP', 'USDC', 'ADA', 'STETH', 'AVAX', 'DOGE', 'WTRX', 'TRX', 'DOT', 'LINK', 'TON11419', 'MATIC', 'WBTC', 'SHIB', 'DAI', 'LTC', 'WEOS', 'BCH', 'ICP', 'ATOM'], index=0)

    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-12-21'))

    st.subheader('Today\'s Data for BTC')

    news = get_today_cryptonews()

    with st.expander('📰 Crypto News'):
        for article in news:
            colA, colB = st.columns([1, 3])
            try:
                colA.image(article['urlToImage'], use_column_width=True)
            except:
                continue
            colB.markdown(f'''
                ### {article['title']}
                {article['content']}
                [Read more]({article['url']})
            ''')

    today_df = get_today_crypto_data()
    st.write(today_df)
    colA, colB, colC = st.columns(3)
    colA.metric(label='Open', value=round(today_df['Open'][0], 2), delta=round(today_df['Change'][0], 2))
    colB.metric(label='Close', value=round(today_df['Close'][0], 2), delta=round(today_df['Change'][0], 2))
    colC.metric(label='Volume', value=round(today_df['Volume'][0], 2), delta='in dollars ($)', delta_color='normal')

    today_cryptonews_sentiment = get_today_cryptonews_sentiment(news)
    st.write(today_cryptonews_sentiment)
    predicted_price = get_prediction(today_cryptonews_sentiment, today_df['Close'][0])
    st.info(f'Predicted BTC price in USD for tomorrow: {predicted_price} $')

    st.write('---')

    crypto_df = fetch_crypto_data(selected_crypto, start_date, end_date)

    st.subheader('Data from `' + start_date.strftime('%Y-%m-%d') + '` to `' + end_date.strftime('%Y-%m-%d') + '` for `' + selected_crypto + '` in `USD ($)`')
    st.dataframe(crypto_df.style.highlight_max(axis=0, color='green').highlight_min(axis=0, color='purple'))

    st.subheader('Opening and Closing Price')
    st.line_chart(crypto_df[['Open', 'Close']])

    st.subheader('Volume')
    st.line_chart(crypto_df['Volume'])

    st.subheader('Volatility')
    st.line_chart(crypto_df['High'] - crypto_df['Low'])

    st.sidebar.info('Oraclus - CryptoPappy. Cryptocurrency Analysis App based on [Streamlit](https://streamlit.io) framework.')
    st.sidebar.caption('Made with ❤️ by [DataFlow](https://dataflow.kz) team.')

if __name__ == "__main__":
    main()
