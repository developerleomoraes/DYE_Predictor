# == ================================================ == #
# == CREATE BACKPROPAGATION MODEL                     == #
# == ================================================ == #


## == Import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from datetime import datetime, timedelta



class BP_ForexPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.forecast_dates = None
        self.forecast_values = None
        self.features = ['exchange_rate_USD_EUR', 'exchange_rate_JPY_EUR', 
                        'oil_price', 'sp500', 'interest_rate',
                        'day_of_week', 'month', 'year']
        self.targets = ['exchange_rate_USD_EUR', 'exchange_rate_JPY_EUR']
        
    def load_and_preprocess_data(self):
        """Carrega e pré-processa os dados"""
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        
        X = df[self.features].values
        y = df[self.targets].values
        
        ## == Normalize
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        ## == 80% training | 20% Test
        split_idx = int(len(X_scaled) * 0.8)
        return (X_scaled[:split_idx], y_scaled[:split_idx], 
                X_scaled[split_idx:], y_scaled[split_idx:], df)
    
    

    def build_model(self, input_shape):
        """Constrói o modelo de rede neural com backpropagation"""
        model = Sequential([
            Dense(128, input_dim=input_shape, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(len(self.targets), activation='linear')
        ])
        
        optimizer = Adam(learning_rate=0.0005)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        return model
    

    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Treina o modelo com early stopping"""
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
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

        X_forecast = []
        
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_test)
        
        ## == column
        rmse_usd = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        rmse_jpy = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
        
        ## == 5 days
        last_dates = [df['date'].iloc[-1] + timedelta(days=i) for i in range(1, 6)]
        last_features = df[self.features].iloc[-1:].values
        
        
        for i in range(1, 6):
            
            new_row = last_features.copy()
            
            future_date = df['date'].iloc[-1] + timedelta(days=i)
            new_row[0, self.features.index('day_of_week')] = future_date.weekday()
            new_row[0, self.features.index('month')] = future_date.month
            new_row[0, self.features.index('year')] = future_date.year
            
            X_forecast.append(new_row)
        
        X_forecast = np.concatenate(X_forecast, axis=0)
        X_forecast_scaled = self.scaler_X.transform(X_forecast)
        
        ## == Predictions
        y_forecast_scaled = self.model.predict(X_forecast_scaled)
        y_forecast = self.scaler_y.inverse_transform(y_forecast_scaled)
        
        self.forecast_dates = last_dates
        self.forecast_values = y_forecast
        
        return rmse_usd, rmse_jpy
    




    def plot_results(self, df):
        """Visualiza os resultados com histórico completo e previsões"""
        plt.figure(figsize=(16, 12))
        
        ## == Transform data
        X_full = self.scaler_X.transform(df[self.features].values)
        y_pred_full_scaled = self.model.predict(X_full)
        y_pred_full = self.scaler_y.inverse_transform(y_pred_full_scaled)
        
        plot_df = df.copy()
        plot_df['USD_EUR_pred'] = y_pred_full[:, 0]
        plot_df['JPY_EUR_pred'] = y_pred_full[:, 1]
        
        ## == df
        future_df = pd.DataFrame({
            'Data': self.forecast_dates,
            'USD/EUR_Previsto': self.forecast_values[:, 0],
            'JPY/EUR_Previsto': self.forecast_values[:, 1]
        })
        
        ## == Future predictions
        print("\nPrevisões para os próximos 5 dias:")
        print(future_df.to_string(index=False, float_format="%.6f"))
        print("\n" + "="*70 + "\n")
        
        ## == USD/EUR
        plt.subplot(2, 1, 1)
        
        ## == Historical
        plt.plot(plot_df['date'], plot_df['exchange_rate_USD_EUR'], 
                label='Valor Real', color='#1f77b4', linewidth=2, alpha=0.9)
        
        ## == Model prediction
        plt.plot(plot_df['date'], plot_df['USD_EUR_pred'], 
                label='Previsão do Modelo', color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ## 5 days
        plt.plot(future_df['Data'], future_df['USD/EUR_Previsto'], 
                'ro-', markersize=8, linewidth=2, label='Previsão Futura (5 dias)')
        
        ## ==histórico/futuro
        last_date = plot_df['date'].iloc[-1]
        plt.axvline(x=last_date, color='gray', linestyle=':', linewidth=1)
        plt.text(last_date, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
                ' Fim dos Dados Históricos', ha='left', va='bottom', color='gray')
        
        plt.title('USD/EUR - Valores Reais vs Previsões', fontsize=14, pad=20)
        plt.ylabel('Taxa de Câmbio', fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        ## == JPY/EUR
        plt.subplot(2, 1, 2)
        
        ## == Historical
        plt.plot(plot_df['date'], plot_df['exchange_rate_JPY_EUR'], 
                label='Valor Real', color='#2ca02c', linewidth=2, alpha=0.9)
        
        ## Model predictions
        plt.plot(plot_df['date'], plot_df['JPY_EUR_pred'], 
                label='Previsão do Modelo', color='#9467bd', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ## future predictions
        plt.plot(future_df['Data'], future_df['JPY/EUR_Previsto'], 
                'mo-', markersize=8, linewidth=2, label='Previsão Futura (5 dias)')
        
        ## == historical/future
        plt.axvline(x=last_date, color='gray', linestyle=':', linewidth=1)
        plt.text(last_date, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
                ' Fim dos Dados Históricos', ha='left', va='bottom', color='gray')
        
        plt.title('JPY/EUR - Valores Reais vs Previsões', fontsize=14, pad=20)
        plt.ylabel('Taxa de Câmbio', fontsize=12)
        plt.xlabel('Data', fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        return future_df

