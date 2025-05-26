
# == ================================================ == #
# == CREATE LSTM MODEL                                == #
# == ================================================ == #



## Import Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam



class ForexPredictorLSTM:
    def __init__(self, data_path, look_back=30):
        self.data_path = data_path
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.forecast_dates = None
        self.forecast_values = None
        self.look_back = look_back  
        self.features = ['exchange_rate_USD_EUR', 'exchange_rate_JPY_EUR', 
                        'oil_price', 'sp500', 'interest_rate',
                        'day_of_week', 'month', 'year']
        self.targets = ['exchange_rate_USD_EUR', 'exchange_rate_JPY_EUR']
        
    def load_and_preprocess_data(self):
        """Carrega e pré-processa os dados para LSTM"""
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        X = df[self.features].values
        y = df[self.targets].values
        
        ## == normalize data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        ## == LSTM ([samples, time steps, features])
        X_lstm, y_lstm = self.create_dataset(X_scaled, y_scaled)
        
        ## ==  80% training | 20% test
        split_idx = int(len(X_lstm) * 0.8)
        return (X_lstm[:split_idx], y_lstm[:split_idx], 
                X_lstm[split_idx:], y_lstm[split_idx:], df)
    

    def create_dataset(self, X, y):
        """Cria o dataset no formato adequado para LSTM"""
        X_lstm, y_lstm = [], []
        for i in range(self.look_back, len(X)):
            X_lstm.append(X[i-self.look_back:i, :])
            y_lstm.append(y[i, :])
        return np.array(X_lstm), np.array(y_lstm)
    

    def build_model(self, input_shape):
        """Constrói o modelo LSTM"""
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(32, activation='relu'),
            Dense(len(self.targets), activation='linear')
        ])
        
        optimizer = Adam(learning_rate=0.0005)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        return model
    

    def train_model(self, X_train, y_train, X_test, y_test):
        """Treina o modelo com early stopping"""
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[early_stop]
        )
        return history
    

    

    def make_predictions(self, X_test, y_test, df):
        """Faz previsões e avalia o modelo"""
       
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_test)
        
        ## =RMSE
        rmse_usd = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        rmse_jpy = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
        
        ## == 5 days
        last_dates = [df['date'].iloc[-1] + timedelta(days=i) for i in range(1, 6)]
        
        ## == look_back
        last_features = df[self.features].values[-self.look_back:]
        X_forecast = self.scaler_X.transform(last_features)
        X_forecast = X_forecast.reshape(1, self.look_back, len(self.features))
        
        ## == Predictions
        y_forecast = []
        current_input = X_forecast.copy()
        
        for i in range(5):
          
            pred_scaled = self.model.predict(current_input)[0]
            pred = self.scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
            y_forecast.append(pred)
            
           
            new_features = last_features[-1].copy()
            new_features[0] = pred[0]  # USD/EUR prediction
            new_features[1] = pred[1]  # JPY/EUR prediction
            
            
            future_date = df['date'].iloc[-1] + timedelta(days=i+1)
            new_features[self.features.index('day_of_week')] = future_date.weekday()
            new_features[self.features.index('month')] = future_date.month
            new_features[self.features.index('year')] = future_date.year
            
            # update last_features
            last_features = np.vstack([last_features[1:], new_features])
            
            ## == update current_input
            new_input = self.scaler_X.transform(new_features.reshape(1, -1))
            current_input = np.concatenate([
                current_input[:, 1:, :], 
                new_input.reshape(1, 1, -1)
            ], axis=1)
        
        self.forecast_dates = last_dates
        self.forecast_values = np.array(y_forecast)
        
        return rmse_usd, rmse_jpy




    
    def plot_results(self, df):
        """Visualiza os resultados com histórico completo e previsões"""
        plt.figure(figsize=(16, 12))
        
        X_full = self.scaler_X.transform(df[self.features].values)
        X_full_lstm, y_full_lstm = self.create_dataset(X_full, 
                                                     self.scaler_y.transform(df[self.targets].values))
        
        y_pred_full_scaled = self.model.predict(X_full_lstm)
        y_pred_full = self.scaler_y.inverse_transform(y_pred_full_scaled)
        
        plot_dates = df['date'].iloc[self.look_back:].reset_index(drop=True)
        plot_df = df.iloc[self.look_back:].copy()
        plot_df['USD_EUR_pred'] = y_pred_full[:, 0]
        plot_df['JPY_EUR_pred'] = y_pred_full[:, 1]
        
        ## == df
        future_df = pd.DataFrame({
            'Data': self.forecast_dates,
            'USD/EUR_Previsto': self.forecast_values[:, 0],
            'JPY/EUR_Previsto': self.forecast_values[:, 1]
        })
        
        ## == 5 days predictions
        print("\nPrevisões para os próximos 5 dias:")
        print(future_df.to_string(index=False, float_format="%.6f"))
        print("\n" + "="*70 + "\n")
        
        ## == USD/EUR
        plt.subplot(2, 1, 1)
        
        ## == historical
        plt.plot(plot_df['date'], plot_df['exchange_rate_USD_EUR'], 
                label='Valor Real', color='#1f77b4', linewidth=2, alpha=0.9)
        
        ## == models predictions
        plt.plot(plot_df['date'], plot_df['USD_EUR_pred'], 
                label='Previsão do Modelo', color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ## == future predictions
        plt.plot(future_df['Data'], future_df['USD/EUR_Previsto'], 
                'ro-', markersize=8, linewidth=2, label='Previsão Futura (5 dias)')
        
        ## == historical/future
        last_date = plot_df['date'].iloc[-1]
        plt.axvline(x=last_date, color='gray', linestyle=':', linewidth=1)
        plt.text(last_date, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
                ' Fim dos Dados Históricos', ha='left', va='bottom', color='gray')
        
        plt.title('USD/EUR - Valores Reais vs Previsões (LSTM)', fontsize=14, pad=20)
        plt.ylabel('Taxa de Câmbio', fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        ## ==JPY/EUR
        plt.subplot(2, 1, 2)
        
        ## == historical
        plt.plot(plot_df['date'], plot_df['exchange_rate_JPY_EUR'], 
                label='Valor Real', color='#2ca02c', linewidth=2, alpha=0.9)
        
        ## == model predictions
        plt.plot(plot_df['date'], plot_df['JPY_EUR_pred'], 
                label='Previsão do Modelo', color='#9467bd', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ## 5 days predictions
        plt.plot(future_df['Data'], future_df['JPY/EUR_Previsto'], 
                'mo-', markersize=8, linewidth=2, label='Previsão Futura (5 dias)')
        
        ## == historical/future
        plt.axvline(x=last_date, color='gray', linestyle=':', linewidth=1)
        plt.text(last_date, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
                ' Fim dos Dados Históricos', ha='left', va='bottom', color='gray')
        
        plt.title('JPY/EUR - Valores Reais vs Previsões (LSTM)', fontsize=14, pad=20)
        plt.ylabel('Taxa de Câmbio', fontsize=12)
        plt.xlabel('Data', fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        return future_df
    


