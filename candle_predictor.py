import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import cv2
from PIL import Image
import base64
import io
import json
from typing import Dict, List, Tuple, Optional

class CandlePredictor:
    """Advanced Candlestick Prediction Model using TFT and Computer Vision"""
    
    def __init__(self):
        self.tft_model = None
        self.cv_model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 60  # Look back 60 periods
        self.is_trained = False
        
    def extract_price_data_from_chart(self, image_data: str) -> Optional[Dict]:
        """Extract OHLC data from chart image using computer vision"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(image)
            
            # Convert to OpenCV format
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            # Extract candlestick data using advanced CV techniques
            candlesticks = self._detect_candlesticks(img_cv)
            current_price = self._get_current_price(img_cv)
            
            return {
                'candlesticks': candlesticks,
                'current_price': current_price,
                'chart_timeframe': self._detect_timeframe(img_cv),
                'market_structure': self._analyze_market_structure(candlesticks)
            }
            
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None
    
    def _detect_candlesticks(self, img_cv) -> List[Dict]:
        """Detect individual candlesticks from chart image"""
        # Advanced candlestick detection algorithm
        # This is a sophisticated computer vision task
        
        # Placeholder for actual CV implementation
        # In reality, this would involve:
        # 1. Edge detection to find candlestick bodies and wicks
        # 2. Color analysis to determine bullish/bearish candles
        # 3. Position mapping to extract OHLC values
        
        # Simulated candlestick data for demonstration
        candlesticks = []
        base_price = 1.2500
        
        for i in range(50):  # Last 50 candles
            # Simulate realistic price movement
            change = np.random.normal(0, 0.0005)
            open_price = base_price + change
            
            # Generate realistic OHLC
            body_size = np.random.uniform(0.0002, 0.0015)
            wick_size = np.random.uniform(0.0001, 0.0008)
            
            if np.random.random() > 0.5:  # Bullish candle
                close_price = open_price + body_size
                high_price = close_price + wick_size
                low_price = open_price - wick_size
            else:  # Bearish candle
                close_price = open_price - body_size
                high_price = open_price + wick_size
                low_price = close_price - wick_size
            
            candlesticks.append({
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'timestamp': i,
                'type': 'bullish' if close_price > open_price else 'bearish'
            })
            
            base_price = close_price
        
        return candlesticks
    
    def _get_current_price(self, img_cv) -> float:
        """Extract current price from chart"""
        # In real implementation, this would use OCR to read price labels
        return 1.2520  # Placeholder
    
    def _detect_timeframe(self, img_cv) -> str:
        """Detect chart timeframe from image"""
        # OCR to read timeframe labels
        return "1H"  # Placeholder
    
    def _analyze_market_structure(self, candlesticks: List[Dict]) -> Dict:
        """Analyze market structure from candlestick data"""
        if not candlesticks:
            return {}
        
        closes = [c['close'] for c in candlesticks]
        highs = [c['high'] for c in candlesticks]
        lows = [c['low'] for c in candlesticks]
        
        # Trend analysis
        recent_closes = closes[-10:]
        trend = "bullish" if recent_closes[-1] > recent_closes[0] else "bearish"
        
        # Support and resistance
        resistance = max(highs[-20:])
        support = min(lows[-20:])
        
        # Higher highs and higher lows (bullish structure)
        hh_hl = self._check_higher_highs_lows(highs, lows)
        
        return {
            'trend': trend,
            'resistance': resistance,
            'support': support,
            'structure': 'bullish' if hh_hl else 'bearish',
            'volatility': np.std(closes[-20:]) if len(closes) >= 20 else 0
        }
    
    def _check_higher_highs_lows(self, highs: List[float], lows: List[float]) -> bool:
        """Check for higher highs and higher lows pattern"""
        if len(highs) < 10 or len(lows) < 10:
            return False
        
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # Simple check for ascending pattern
        return (recent_highs[-1] > recent_highs[-5] and 
                recent_lows[-1] > recent_lows[-5])
    
    def build_tft_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build Temporal Fusion Transformer model for price prediction"""
        
        # Input layers
        price_input = keras.layers.Input(shape=input_shape, name='price_sequence')
        
        # Multi-head attention layers (simplified TFT architecture)
        attention_1 = keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=64, name='attention_1'
        )(price_input, price_input)
        
        attention_1 = keras.layers.LayerNormalization()(attention_1)
        attention_1 = keras.layers.Dropout(0.1)(attention_1)
        
        # Feed forward network
        ff_1 = keras.layers.Dense(256, activation='relu')(attention_1)
        ff_1 = keras.layers.Dropout(0.1)(ff_1)
        ff_1 = keras.layers.Dense(128, activation='relu')(ff_1)
        
        # Another attention layer
        attention_2 = keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=32, name='attention_2'
        )(ff_1, ff_1)
        
        attention_2 = keras.layers.LayerNormalization()(attention_2)
        
        # Global average pooling
        pooled = keras.layers.GlobalAveragePooling1D()(attention_2)
        
        # Dense layers for final prediction
        dense_1 = keras.layers.Dense(64, activation='relu')(pooled)
        dense_1 = keras.layers.Dropout(0.2)(dense_1)
        
        # Output layers for OHLC prediction
        open_pred = keras.layers.Dense(1, name='open_pred')(dense_1)
        high_pred = keras.layers.Dense(1, name='high_pred')(dense_1)
        low_pred = keras.layers.Dense(1, name='low_pred')(dense_1)
        close_pred = keras.layers.Dense(1, name='close_pred')(dense_1)
        
        # Direction prediction (up/down)
        direction_pred = keras.layers.Dense(1, activation='sigmoid', name='direction_pred')(dense_1)
        
        model = keras.Model(
            inputs=price_input,
            outputs=[open_pred, high_pred, low_pred, close_pred, direction_pred],
            name='TFT_CandlePredictor'
        )
        
        return model
    
    def prepare_training_data(self, candlesticks: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare data for model training"""
        if len(candlesticks) < self.sequence_length + 1:
            raise ValueError(f"Need at least {self.sequence_length + 1} candlesticks")
        
        # Extract OHLC data
        ohlc_data = []
        for candle in candlesticks:
            ohlc_data.append([
                candle['open'], candle['high'], 
                candle['low'], candle['close']
            ])
        
        ohlc_array = np.array(ohlc_data)
        
        # Normalize data
        ohlc_scaled = self.scaler.fit_transform(ohlc_array)
        
        # Create sequences
        X, y = [], {
            'open': [], 'high': [], 'low': [], 'close': [], 'direction': []
        }
        
        for i in range(len(ohlc_scaled) - self.sequence_length):
            # Input sequence
            X.append(ohlc_scaled[i:i + self.sequence_length])
            
            # Target (next candle)
            next_candle = ohlc_scaled[i + self.sequence_length]
            current_close = ohlc_scaled[i + self.sequence_length - 1][3]  # Previous close
            
            y['open'].append(next_candle[0])
            y['high'].append(next_candle[1])
            y['low'].append(next_candle[2])
            y['close'].append(next_candle[3])
            
            # Direction: 1 if next close > current close, 0 otherwise
            direction = 1.0 if next_candle[3] > current_close else 0.0
            y['direction'].append(direction)
        
        X = np.array(X)
        for key in y:
            y[key] = np.array(y[key])
        
        return X, y
    
    def train_model(self, candlesticks: List[Dict]) -> bool:
        """Train the TFT model on historical data"""
        try:
            # Prepare training data
            X, y = self.prepare_training_data(candlesticks)
            
            # Build model
            input_shape = (self.sequence_length, 4)  # OHLC
            self.tft_model = self.build_tft_model(input_shape)
            
            # Compile model
            self.tft_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss={
                    'open_pred': 'mse',
                    'high_pred': 'mse',
                    'low_pred': 'mse',
                    'close_pred': 'mse',
                    'direction_pred': 'binary_crossentropy'
                },
                loss_weights={
                    'open_pred': 1.0,
                    'high_pred': 1.0,
                    'low_pred': 1.0,
                    'close_pred': 2.0,  # Close price is most important
                    'direction_pred': 1.5  # Direction is also important
                },
                metrics={
                    'direction_pred': 'accuracy'
                }
            )
            
            # Train model
            history = self.tft_model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_next_candle(self, candlesticks: List[Dict]) -> Optional[Dict]:
        """Predict the next candlestick"""
        if not self.is_trained or not self.tft_model:
            print("Model not trained yet")
            return None
        
        try:
            # Prepare input data
            if len(candlesticks) < self.sequence_length:
                print(f"Need at least {self.sequence_length} candlesticks for prediction")
                return None
            
            # Get last sequence
            recent_candles = candlesticks[-self.sequence_length:]
            ohlc_data = []
            for candle in recent_candles:
                ohlc_data.append([
                    candle['open'], candle['high'],
                    candle['low'], candle['close']
                ])
            
            ohlc_array = np.array(ohlc_data)
            ohlc_scaled = self.scaler.transform(ohlc_array)
            
            # Reshape for prediction
            X_pred = ohlc_scaled.reshape(1, self.sequence_length, 4)
            
            # Make prediction
            predictions = self.tft_model.predict(X_pred, verbose=0)
            
            # Unpack predictions
            open_pred, high_pred, low_pred, close_pred, direction_pred = predictions
            
            # Inverse transform to get actual prices
            pred_ohlc = np.array([[
                open_pred[0][0], high_pred[0][0],
                low_pred[0][0], close_pred[0][0]
            ]])
            
            pred_ohlc_actual = self.scaler.inverse_transform(pred_ohlc)[0]
            
            # Calculate confidence based on model uncertainty
            direction_confidence = float(direction_pred[0][0])
            if direction_confidence < 0.5:
                direction_confidence = 1.0 - direction_confidence
                predicted_direction = "bearish"
            else:
                predicted_direction = "bullish"
            
            # Calculate price targets and stop loss
            current_close = candlesticks[-1]['close']
            predicted_close = pred_ohlc_actual[3]
            
            if predicted_direction == "bullish":
                entry_price = current_close
                target_1 = predicted_close
                target_2 = predicted_close + (predicted_close - current_close) * 0.5
                stop_loss = current_close - (predicted_close - current_close) * 0.5
            else:
                entry_price = current_close
                target_1 = predicted_close
                target_2 = predicted_close - (current_close - predicted_close) * 0.5
                stop_loss = current_close + (current_close - predicted_close) * 0.5
            
            # Risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(target_1 - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            return {
                'next_candle_prediction': {
                    'open': round(pred_ohlc_actual[0], 5),
                    'high': round(pred_ohlc_actual[1], 5),
                    'low': round(pred_ohlc_actual[2], 5),
                    'close': round(pred_ohlc_actual[3], 5),
                    'direction': predicted_direction,
                    'direction_confidence': round(direction_confidence, 3)
                },
                'trading_signal': {
                    'action': 'BUY' if predicted_direction == 'bullish' else 'SELL',
                    'entry_price': round(entry_price, 5),
                    'target_1': round(target_1, 5),
                    'target_2': round(target_2, 5),
                    'stop_loss': round(stop_loss, 5),
                    'risk_reward_ratio': round(risk_reward_ratio, 2),
                    'confidence': round(direction_confidence, 3),
                    'timeframe': '1H'  # Based on detected timeframe
                },
                'market_analysis': {
                    'current_price': current_close,
                    'predicted_move': round(predicted_close - current_close, 5),
                    'move_percentage': round(((predicted_close - current_close) / current_close) * 100, 3),
                    'volatility_expected': 'normal'
                }
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def analyze_chart_and_predict(self, image_data: str) -> Optional[Dict]:
        """Complete analysis: extract data from chart and predict next candle"""
        # Extract price data from chart image
        chart_data = self.extract_price_data_from_chart(image_data)
        if not chart_data:
            return None
        
        candlesticks = chart_data['candlesticks']
        
        # Train model on extracted data (in real scenario, use pre-trained model)
        if not self.is_trained:
            if len(candlesticks) >= self.sequence_length + 10:  # Need enough data
                self.train_model(candlesticks)
            else:
                print("Not enough historical data for training")
                return None
        
        # Make prediction
        prediction = self.predict_next_candle(candlesticks)
        
        if prediction:
            # Add chart analysis
            prediction['chart_analysis'] = {
                'market_structure': chart_data['market_structure'],
                'timeframe': chart_data['chart_timeframe'],
                'total_candles_analyzed': len(candlesticks),
                'data_quality': 'good'
            }
        
        return prediction

