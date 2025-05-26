# == ================================================ == #
# == =========== GET VARIABLES FROM API ============= == #
# == ================================================ == #

## == Import libs
import pandas as pd
import numpy as np
import requests
import dotenv
import os
from datetime import datetime
import matplotlib.pyplot as plt




## == =============== ALPHA VANTAGE API =============== == ##
## Função para obter dados de câmbio usando Alpha Vantage
def get_exchange_rate_data(start_date: str, end_date: str, from_currency: str, to_currency: str, ALPHA_VANTAGE_API_KEY: str) -> pd.DataFrame:

    dates = []
    rates = []
    
    print(f"    ↳ Obtendo dados de câmbio {from_currency}/{to_currency} de {start_date} a {end_date}...")
    
    try:
        alpha_vantage_url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}&datatype=json"

        response = requests.get(alpha_vantage_url)
        data = response.json()
    
        
        for date, values in data['Time Series FX (Daily)'].items():
            if date >= start_date and date <= end_date:
                dates.append(date)
                rates.append(float(values['4. close']))
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            f'exchange_rate_{from_currency}_{to_currency}': rates
        })
        
        return df.sort_values('date').reset_index(drop=True)
    
    except Exception as e:

        print(f"Erro ao acessar API de câmbio: {e}")
        return pd.DataFrame()
    


## == =============== EIA PETROLEUM API =============== == ##
## == https://www.eia.gov/opendata/
def get_oil_prices(start_date: str, end_date: str, EIA_OIL_API_KEY: str) -> pd.DataFrame:

    print(f"    ↳ Obtendo dados de preço do petróleo de {start_date} a {end_date}...")
    
    try:
        oil_prices_url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
        params = {
            "api_key": EIA_OIL_API_KEY,
            "frequency": "daily",
            "data[0]": "value",
            "start": start_date,
            "end": end_date,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc"
        }
        
        response = requests.request('GET', oil_prices_url, params=params)
        data = response.json()
        
        # Processa os dados
        df = pd.DataFrame(data['response']['data'])
        
        # Limpa e formata os dados
        df = (
            df[['period', 'value']]
            .rename(columns={'period': 'date', 'value': 'oil_price'})
            .assign(
                date=lambda x: pd.to_datetime(x['date']),
                oil_price=lambda x: pd.to_numeric(x['oil_price'])
            )
            .sort_values('date')
            .reset_index(drop=True)
        )
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição HTTP: {e}")
        return pd.DataFrame(columns=['date', 'oil_price'])




## == ==================== FRED API ====================== == ##
# Função para obter dados de taxas de juros (FED Funds Rate)
def get_interest_rates(start_date, end_date, FRED_API_KEY) -> pd.DataFrame:
    print(f"    ↳ Obtendo dados de taxas de juros de {start_date} a {end_date}...")
    
    try:
        series_id = 'DFF'  # FED Funds Rate
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start_date}&observation_end={end_date}"

        response = requests.request('GET', url)
        data = response.json()
        
        df = pd.DataFrame(data['observations'])
        df = df[['date', 'value']].rename(columns={'value': 'interest_rate'})
        df['date'] = pd.to_datetime(df['date'])
        df['interest_rate'] = pd.to_numeric(df['interest_rate'].replace('.', np.nan))
        
        return df.sort_values('date').reset_index(drop=True)
    
    except Exception as e:
        print(f"Erro ao acessar API do FRED: {e}")
        return pd.DataFrame()
    



## == ==================== ALPHA VANTAGE API ====================== == ##

def get_sp500(start_date: str, end_date: str, ALPHA_VANTAGE_API_KEY: str) -> pd.DataFrame:

    dates = []
    closes = []

    print(f"    ↳ Obtendo dados do S&P 500 de {start_date} a {end_date}...")
    
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}&datatype=json"

        response = requests.get(url)
        data = response.json()
        
        for date, values in data['Time Series (Daily)'].items():
            if date >= start_date and date <= end_date:
                dates.append(date)
                closes.append(float(values['4. close']))
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'sp500': closes
        })
        
        return df.sort_values('date').reset_index(drop=True)
    
    except Exception as e:
        print(f"Erro ao acessar API Alpha Vantage: {e}")
        return pd.DataFrame()
    







# == ================================================ == #
# == ================ TRANSFORM DATA ================ == #
# == ================================================ == #


def build_complete_dataset(start_date, end_date) -> pd.DataFrame:
    

    ## == Load globals variables
    dotenv.load_dotenv(dotenv.find_dotenv())

    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    EIA_OIL_API_KEY = os.getenv('EIA_OIL_API_KEY')


    # Obter dados de câmbio para ambos os pares
    usd_eur_df = get_exchange_rate_data(start_date, end_date, 'USD', 'EUR', ALPHA_VANTAGE_API_KEY)
    jpy_eur_df = get_exchange_rate_data(start_date, end_date, 'JPY', 'EUR', ALPHA_VANTAGE_API_KEY)
    
    # Combinar os dados de câmbio primeiro
    df = usd_eur_df.merge(jpy_eur_df, on='date', how='outer')
    
    # Obter outros dados econômicos
    oil_df = get_oil_prices(start_date, end_date, EIA_OIL_API_KEY)
    interest_df = get_interest_rates(start_date, end_date, FRED_API_KEY)
    sp500_df = get_sp500(start_date, end_date, ALPHA_VANTAGE_API_KEY)

    
    # Juntar todos os dataframes
    for data in [oil_df, interest_df, sp500_df]:
        if not data.empty:
            df = df.merge(data, on='date', how='left')
    
    # Preencher valores ausentes (substituição do fillna obsoleto)
    df = df.ffill()  # Preenche para frente
    df = df.bfill()  # Preenche para trás (caso ainda haja NAs no início)
    
    # Adicionar features temporais
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Adicionar diferença percentual diária para todas as taxas de câmbio
    for col in ['exchange_rate_USD_EUR', 'exchange_rate_JPY_EUR', 'oil_price', 'sp500']:
        if col in df.columns:
            df[f'{col}_pct_change'] = df[col].pct_change() * 100
    
    # Adicionar médias móveis
    windows = [7, 30, 90]
    for window in windows:
        for pair in ['USD_EUR', 'JPY_EUR']:
            col = f'exchange_rate_{pair}'
            if col in df.columns:
                df[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
        if 'oil_price' in df.columns:
            df[f'oil_price_ma_{window}'] = df['oil_price'].rolling(window=window).mean()
    
    # Remover linhas com valores NaN
    df.dropna(inplace=True)
    
    return df




