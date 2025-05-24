# == ================================================ == #
# == =========== GET VARIABLES FROM API ============= == #
# == ================================================ == #

## == Import libs
import pandas as pd
import requests
import numpy as np


# == =============== ALPHA VANTAGE API =============== == #
# Função para obter dados de câmbio usando Alpha Vantage
def get_exchange_rate_data(start_date: str, end_date: str, from_currency: str, to_currency: str, ALPHA_VANTAGE_API_KEY: str) -> pd.DataFrame:

    dates = []
    rates = []
    
    print(f"Obtendo dados de câmbio {from_currency}/{to_currency} de {start_date} a {end_date}...")
    
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
    



def get_oil_prices(start_date: str, end_date: str, EIA_OIL_API_KEY: str) -> pd.DataFrame:

    print(f"Obtendo dados de preço do petróleo de {start_date} a {end_date}...")
    
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





# Função para obter dados de taxas de juros (FED Funds Rate)
def get_interest_rates(start_date, end_date, FRED_API_KEY):
    print(f"Obtendo dados de taxas de juros de {start_date} a {end_date}...")
    
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
    