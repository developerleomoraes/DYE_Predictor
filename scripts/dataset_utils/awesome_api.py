# == ================================================ == #
# == Get data from Awesome API                        == #
# == GSHEETS: https://docs.awesomeapi.com.br/         == #
# == ================================================ == #
from cfg.type_mapping import type_mapping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import requests
import time



## == AWESOME API FUNCTIONS =========================================================================
def get_exchange(coin: str, awesome_api_key: str) -> list:

    days_number = 360

    headers = {
    'Accept': 'application/json'
    }

    params = {
        'token': awesome_api_key
    }

  
    ## == DO REQUEST TO API
    url_endpoint = f"https://economia.awesomeapi.com.br/json/daily/{coin}/{days_number}"
    response = requests.request("GET", url_endpoint, headers=headers, params=params)

    print(f'Response req for coin {coin}: {response.status_code}')

    response = response.json()

    return response



def adjuste_null_values(df: pd.DataFrame, resp: list) -> pd.DataFrame:

    df['code'] = resp[0]['code']
    df['codein'] = resp[0]['codein']
    df['name'] = resp[0]['name']
    df['create_date'] = resp[0]['create_date']

    df['timestamp'] = df['timestamp'].apply(lambda epoch_time: time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(float(epoch_time))))

    return df



def adjust_types(df: pd.DataFrame) -> pd.DataFrame:

    for col, dtype in type_mapping.items():
        if dtype == 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(dtype)

    return df
## == AWESOME API FUNCTIONS =========================================================================




