import os
import numpy as np
import pandas as pd
import pyodbc
import json
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import traceback

class TravelRecommendationSystem:
    def __init__(self, db_path=None):
        """Initialize the recommendation system with the database path."""
        # For SQL Server, db_path is ignored and we use a direct connection
        self.model = None
        self.encoders = {}
        self.scaler = None
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            
    def connect_to_db(self):
        """Create a connection to the SQL Server database."""
        conn_str = (
            "DRIVER=/opt/homebrew/lib/libtdsodbc.so;"
            "SERVER=127.0.0.1;"
            "PORT=1433;"
            "DATABASE=virtuoso_travel;"
            "UID=climbing_user;"
            "PWD=hoosierheights;"
            "TDS_Version=7.4;"
        )
        
        return pyodbc.connect(conn_str)
    
    def extract_user_history(self, user_id):
        """
        Extract a user's travel history and preferences from the database.
        
        Args:
            user_id: The ID of the user to extract history for
            
        Returns:
            A dictionary containing the user's history and preferences
        """
        conn = self.connect_to_db()
        
        # Get user demographic info
        user_query = f"""
        SELECT * FROM users WHERE user_id = ?
        """
        user_info = pd.read_sql_query(user_query, conn, params=(user_id,)).iloc[0].to_dict()
        
        # Get user preferences
        preferences_query = f"""
        SELECT * FROM user_preferences WHERE user_id = ?
        """
        try:
            preferences = pd.read_sql_query(preferences_query, conn, params=(user_id,)).iloc[0].to_dict()
            
            # Parse JSON arrays
            if 'preferred_activities' in preferences:
                preferences['preferred_activities'] = json.loads(preferences['preferred_activities'])
            if 'preferred_climates' in preferences:
                preferences['preferred_climates'] = json.loads(preferences['preferred_climates'])
        except:
            preferences = {}
        
        # Get booking history
        bookings_query = f"""
        SELECT b.booking_id, b.booking_date, b.total_cost, b.booking_status,
               h.hotel_id, h.destination_id as hotel_destination_id, h.star_rating, h.price_category,
               t.tour_id, t.destination_id as tour_destination_id, t.category as tour_category
        FROM bookings b
        LEFT JOIN hotel_bookings hb ON b.booking_id = hb.booking_id
        LEFT JOIN hotels h ON hb.hotel_id = h.hotel_id
        LEFT JOIN tour_bookings tb ON b.booking_id = tb.booking_id
        LEFT JOIN tours t ON tb.tour_id = t.tour_id
        WHERE b.user_id = ? AND b.booking_status = 'completed'
        """
        bookings = pd.read_sql_query(bookings_query, conn, params=(user_id,))
        
        # Get interaction history
        interactions_query = f"""
        SELECT * FROM user_interactions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        """
        interactions = pd.read_sql_query(interactions_query, conn, params=(user_id,))
        
        # Get reviews
        reviews_query = f"""
        SELECT * FROM reviews
        WHERE user_id = ?
        """
        reviews = pd.read_sql_query(reviews_query, conn, params=(user_id,))
        
        conn.close()
        
        return {
            'user_info': user_info,
            'preferences': preferences,
            'bookings': bookings,
            'interactions': interactions,
            'reviews': reviews
        }
    
    def collect_training_data(self):
        """
        Collect and prepare training data for the recommendation model.
        
        This function will:
        1. Extract user interactions and bookings
        2. Create features from user demographics, preferences and history
        3. Create target labels (destinations users have visited/liked)
        
        Returns:
            X: Feature dataframe
            y: Target dataframe (binary indicators for liked destinations)
        """
        conn = self.connect_to_db()
        
        # Get all users
        users = pd.read_sql_query("SELECT * FROM users", conn)
        
        # Get all destinations
        destinations = pd.read_sql_query("SELECT * FROM destinations", conn)
        
        # Get user preferences
        preferences = pd.read_sql_query("SELECT * FROM user_preferences", conn)
        
        # Get completed bookings with destination info
        bookings_query = """
        SELECT b.booking_id, b.user_id, b.booking_date, 
               h.destination_id as hotel_destination_id,
               t.destination_id as tour_destination_id
        FROM bookings b
        LEFT JOIN hotel_bookings hb ON b.booking_id = hb.booking_id
        LEFT JOIN hotels h ON hb.hotel_id = h.hotel_id
        LEFT JOIN tour_bookings tb ON b.booking_id = tb.booking_id
        LEFT JOIN tours t ON tb.tour_id = t.tour_id
        WHERE b.booking_status = 'completed'
        """
        bookings = pd.read_sql_query(bookings_query, conn)
        
        # Get reviews
        reviews = pd.read_sql_query("""
        SELECT r.*, 
               CASE WHEN r.entity_type = 'hotel' THEN h.destination_id
                    WHEN r.entity_type = 'tour' THEN t.destination_id
                    ELSE NULL
               END as destination_id
        FROM reviews r
        LEFT JOIN hotels h ON r.entity_type = 'hotel' AND r.entity_id = h.hotel_id
        LEFT JOIN tours t ON r.entity_type = 'tour' AND r.entity_id = t.tour_id
        WHERE r.rating >= 4
        """, conn)
        
        # Get interaction data
        interactions = pd.read_sql_query("""
        SELECT * FROM user_interactions
        WHERE entity_type = 'destination'
        """, conn)
        
        conn.close()
        
        # Extract destination IDs from bookings
        hotel_destinations = bookings[['user_id', 'hotel_destination_id']].dropna()
        hotel_destinations.rename(columns={'hotel_destination_id': 'destination_id'}, inplace=True)
        
        tour_destinations = bookings[['user_id', 'tour_destination_id']].dropna()
        tour_destinations.rename(columns={'tour_destination_id': 'destination_id'}, inplace=True)
        
        # Combine all visited/liked destinations
        visited_destinations = pd.concat([
            hotel_destinations,
            tour_destinations,
            reviews[['user_id', 'destination_id']].dropna(),
            interactions[interactions['interaction_type'].isin(['view', 'save'])][['user_id', 'entity_id']].rename(columns={'entity_id': 'destination_id'})
        ])
        visited_destinations['destination_id'] = visited_destinations['destination_id'].astype(int)
        
        # Count visits per user-destination pair
        visit_counts = visited_destinations.groupby(['user_id', 'destination_id']).size().reset_index(name='visit_count')
        
        # Create labels: destinations a user has visited/liked
        # We'll consider it a positive label if visit_count >= 1
        visit_counts['liked'] = 1
        
        # Create user features
        user_features = users.merge(preferences, on='user_id', how='left')
        
        # Parse JSON columns
        for idx, row in user_features.iterrows():
            if pd.notna(row['preferred_activities']):
                try:
                    user_features.at[idx, 'preferred_activities'] = json.loads(row['preferred_activities'])
                except:
                    user_features.at[idx, 'preferred_activities'] = []
            else:
                user_features.at[idx, 'preferred_activities'] = []
                
            if pd.notna(row['preferred_climates']):
                try:
                    user_features.at[idx, 'preferred_climates'] = json.loads(row['preferred_climates'])
                except:
                    user_features.at[idx, 'preferred_climates'] = []
            else:
                user_features.at[idx, 'preferred_climates'] = []
        
        # Prepare features for each user-destination pair
        # For now, we'll focus on:
        # 1. User demographics: age, country, loyalty_tier
        # 2. User preferences: preferred_activities, preferred_climates, budget
        # 3. Destination features: region, country, climate, popularity
        
        # Create cartesian product of all users and destinations for training
        all_user_ids = users['user_id'].unique()
        all_destination_ids = destinations['destination_id'].unique()
        
        # For performance reasons, let's just take a sample
        # In a real system, we'd have more sophisticated sampling
        if len(all_user_ids) > 100:
            all_user_ids = np.random.choice(all_user_ids, 100, replace=False)
        
        pairs = []
        for user_id in all_user_ids:
            for dest_id in all_destination_ids:
                pairs.append((user_id, dest_id))
        
        pairs_df = pd.DataFrame(pairs, columns=['user_id', 'destination_id'])
        
        # Add labels
        pairs_df = pairs_df.merge(visit_counts[['user_id', 'destination_id', 'liked']], 
                                on=['user_id', 'destination_id'], how='left')
        pairs_df['liked'] = pairs_df['liked'].fillna(0)
        
        # Add features
        training_data = pairs_df.merge(user_features, on='user_id', how='left')
        training_data = training_data.merge(destinations, on='destination_id', how='left')
        
        # Create feature for climate match
        def climate_match(row):
            if not isinstance(row['preferred_climates'], list):
                return 0
            if row['climate_type'] in row['preferred_climates']:
                return 1
            return 0
        
        training_data['climate_match'] = training_data.apply(climate_match, axis=1)
        
        # Feature engineering
        # Calculate user age
        current_year = pd.Timestamp.now().year
        training_data['birth_year'] = pd.to_datetime(training_data['date_of_birth']).dt.year
        training_data['age'] = current_year - training_data['birth_year']
        
        # Create features
        features = [
            'age', 'loyalty_tier', 'country_of_residence', 'preferred_hotel_stars',
            'preferred_budget_category', 'travel_style', 'region', 'country',
            'climate_type', 'popularity_score', 'climate_match'
        ]
        
        # Prepare X and y
        X = training_data[features]
        y = training_data['liked']
        
        return X, y
    
    def preprocess_data(self, X, fit=True):
        """
        Preprocess data for model training.
        
        Args:
            X: Feature dataframe
            fit: Whether to fit the encoders/scalers (True) or transform only (False)
            
        Returns:
            Preprocessed features as a numpy array
        """
        X_processed = X.copy()
        
        # Handle categorical features
        cat_features = [
            'loyalty_tier', 'country_of_residence', 'preferred_budget_category',
            'travel_style', 'region', 'country', 'climate_type'
        ]
        
        # Handle numerical features
        num_features = ['age', 'preferred_hotel_stars', 'popularity_score', 'climate_match']
        
        # One-hot encode categorical features
        for feature in cat_features:
            if feature not in X_processed.columns:
                continue
                
            if fit:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(X_processed[[feature]])
                self.encoders[feature] = encoder
            else:
                encoder = self.encoders.get(feature)
                if encoder is None:
                    # If we don't have an encoder for this feature (e.g. in prediction)
                    # create a dummy feature
                    X_processed[f"{feature}_unknown"] = 1
                    continue
                encoded = encoder.transform(X_processed[[feature]])
            
            # Add encoded columns
            feature_names = [f"{feature}_{val}" for val in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_processed.index)
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
        
        # Standard scale numerical features
        if fit:
            self.scaler = StandardScaler()
            X_processed[num_features] = self.scaler.fit_transform(X_processed[num_features])
        else:
            if self.scaler is not None:
                # Handle missing numerical features
                for feature in num_features:
                    if feature not in X_processed.columns:
                        X_processed[feature] = 0
                
                X_processed[num_features] = self.scaler.transform(X_processed[num_features])
        
        # Drop original categorical columns
        X_processed = X_processed.drop(columns=[f for f in cat_features if f in X_processed.columns])
        
        return X_processed
    
    def get_user_features(self, user_id):
        """
        Get user features and preferences for recommendation generation.
        
        Args:
            user_id (int): The ID of the user to get features for
            
        Returns:
            dict: Dictionary containing user features and preferences
        """
        try:
            conn = self.connect_to_db()
            if conn is None:
                print(f"Failed to connect to database in get_user_features for user {user_id}")
                return {}
                
            # Get user demographic info
            user_query = """
            SELECT u.*, 
                   p.preferred_hotel_stars, p.preferred_budget_category, 
                   p.preferred_activities, p.preferred_climates, p.travel_style,
                   p.maximum_flight_duration
            FROM users u
            LEFT JOIN user_preferences p ON u.user_id = p.user_id
            WHERE u.user_id = ?
            """
            
            # Execute query safely
            cursor = conn.cursor()
            cursor.execute(user_query, (user_id,))
            
            user_row = cursor.fetchone()
            if not user_row:
                print(f"User {user_id} not found in database")
                conn.close()
                return {}
                
            # Convert row to dictionary
            columns = [column[0] for column in cursor.description]
            user_info = dict(zip(columns, user_row))
            
            # Parse JSON fields
            if user_info.get('preferred_activities'):
                try:
                    user_info['preferred_activities'] = json.loads(user_info['preferred_activities'])
                except:
                    user_info['preferred_activities'] = []
            else:
                user_info['preferred_activities'] = []
                
            if user_info.get('preferred_climates'):
                try:
                    user_info['preferred_climates'] = json.loads(user_info['preferred_climates'])
                except:
                    user_info['preferred_climates'] = []
            else:
                user_info['preferred_climates'] = []
            
            # Calculate age from date_of_birth if available
            if 'date_of_birth' in user_info and user_info['date_of_birth']:
                from datetime import datetime
                birth_date = user_info['date_of_birth']
                today = datetime.now()
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                user_info['age'] = age
            
            conn.close()
            return user_info
            
        except Exception as e:
            print(f"Error in get_user_features for user {user_id}: {str(e)}")
            traceback.print_exc()
            return {}
    
    def prepare_features_for_prediction(self, destinations_df, user_features):
        """
        Prepare features for making predictions on unvisited destinations.
        
        Args:
            destinations_df (DataFrame): DataFrame of destination information
            user_features (dict): Dictionary of user features
            
        Returns:
            DataFrame: Processed features ready for model prediction
        """
        if destinations_df.empty or not user_features:
            return pd.DataFrame()
            
        try:
            # Create a DataFrame for each destination with user features
            prediction_rows = []
            
            for _, destination in destinations_df.iterrows():
                # Combine user and destination features
                row = {
                    # Don't include user_id and destination_id as they're not used for prediction
                    'age': user_features.get('age', 30),  # Default value if missing
                    'loyalty_tier': user_features.get('loyalty_tier', 'Standard'),
                    'country_of_residence': user_features.get('country_of_residence', 'Unknown'),
                    'preferred_hotel_stars': user_features.get('preferred_hotel_stars', 3),
                    'preferred_budget_category': user_features.get('preferred_budget_category', 'moderate'),
                    'travel_style': user_features.get('travel_style', 'leisure'),
                    'region': destination.get('region', 'Unknown'),
                    'country': destination.get('country', 'Unknown'),
                    'climate_type': destination.get('climate_type', 'Unknown'),
                    'popularity_score': destination.get('popularity_score', 0)
                }
                
                # Calculate climate match
                climate_match = 0
                if 'climate_type' in destination and destination['climate_type'] and 'preferred_climates' in user_features:
                    if destination['climate_type'] in user_features['preferred_climates']:
                        climate_match = 1
                row['climate_match'] = climate_match
                
                prediction_rows.append(row)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(prediction_rows)
            
            # Preprocess features
            processed_features = self.preprocess_data(features_df, fit=False)
            
            # Handle missing values
            processed_features = processed_features.fillna(0)
            
            # Check that processed features has same columns as during training
            if self.model is not None:
                expected_features = self.model.feature_names_in_
                missing_features = [f for f in expected_features if f not in processed_features.columns]
                extra_features = [f for f in processed_features.columns if f not in expected_features]
                
                if missing_features:
                    print(f"Warning: Missing expected features: {missing_features}")
                    # Add missing features with zeros
                    for feature in missing_features:
                        processed_features[feature] = 0
                
                if extra_features:
                    print(f"Warning: Extra features not used for training: {extra_features}")
                    # Remove extra features
                    processed_features = processed_features.drop(columns=extra_features)
                
                # Ensure columns are in the same order as the model expects
                processed_features = processed_features[expected_features]
            
            return processed_features
            
        except Exception as e:
            print(f"Error preparing features for prediction: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def clear_model(self):
        """Clear the current model, encoders, and scaler to force retraining or reloading."""
        self.model = None
        self.encoders = {}
        self.scaler = None
        print("Model cleared. Will be retrained or reloaded on next use.")
        
    def train_model(self):
        """Train the XGBoost recommendation model."""
        try:
            print("Collecting training data...")
            X, y = self.collect_training_data()
            
            if X.empty or len(y) == 0:
                print("Error: No training data collected")
                return False
                
            print(f"Data collected: {X.shape[0]} user-destination pairs")
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            print("Preprocessing data...")
            X_train_processed = self.preprocess_data(X_train, fit=True)
            X_test_processed = self.preprocess_data(X_test, fit=False)
            
            # Handle missing values
            X_train_processed = X_train_processed.fillna(0)
            X_test_processed = X_test_processed.fillna(0)
            
            # Keep only common columns between train and test
            common_cols = set(X_train_processed.columns) & set(X_test_processed.columns)
            X_train_processed = X_train_processed[list(common_cols)]
            X_test_processed = X_test_processed[list(common_cols)]
            
            print(f"Processed features: {X_train_processed.shape[1]}")
            
            # Get numeric feature names (after preprocessing)
            feature_names = X_train_processed.columns.tolist()
            
            # Configure XGBoost model
            print("Training XGBoost model...")
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                learning_rate=0.1,
                n_estimators=100,
                max_depth=5,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            # Train model
            self.model.fit(X_train_processed, y_train)
            
            # Store feature names explicitly for later validation
            if not hasattr(self.model, 'feature_names_in_'):
                setattr(self.model, 'feature_names_in_', X_train_processed.columns.tolist())
            
            # Evaluate model
            y_pred = self.model.predict(X_test_processed)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"Model trained. Performance metrics:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Save model and preprocessors
            print("Saving model and preprocessors...")
            with open('models/recommendation_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
                
            with open('models/encoders.pkl', 'wb') as f:
                pickle.dump(self.encoders, f)
                
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
                
            print("Model training complete!")
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            traceback.print_exc()
            self.model = None  # Reset the model
            return False
    
    def load_model(self):
        """Load a trained model and preprocessors from disk."""
        model_path = 'models/recommendation_model.pkl'
        encoders_path = 'models/encoders.pkl'
        scaler_path = 'models/scaler.pkl'
        
        if not (os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(scaler_path)):
            print("Trained model not found. Please train the model first.")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            with open(encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
                
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            # Verify that the model has feature_names_in_ attribute
            if not hasattr(self.model, 'feature_names_in_'):
                print("Warning: Loaded model doesn't have feature_names_in_ attribute. This may cause prediction issues.")
                
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            return False
            
    def train_or_load_model(self):
        """Train a new model or load an existing one if available."""
        if self.model is not None:
            # Model already loaded
            return True
            
        # Try to load existing model
        if os.path.exists('models/recommendation_model.pkl'):
            print("Found existing model, attempting to load it...")
            if self.load_model():
                return True
        
        # If loading failed or no model exists, train a new one
        print("No valid model found. Training a new model...")
        self.train_model()
        return self.model is not None
    
    def fallback_recommendations(self, user_id, num_recommendations=10):
        """
        Generate destination recommendations without using the ML model.
        This is a fallback method when the model fails or is not available.
        
        Args:
            user_id (int): The ID of the user to generate recommendations for
            num_recommendations (int): The number of recommendations to generate
            
        Returns:
            pd.DataFrame: DataFrame of recommended destinations
        """
        try:
            print(f"Using fallback recommendation method for user {user_id}")
            # Connect to database
            conn = self.connect_to_db()
            if conn is None:
                print("Failed to connect to database")
                return pd.DataFrame()
            
            # Get user preferences to use for basic filtering
            user_features = self.get_user_features(user_id)
            
            # Get destinations the user hasn't visited 
            unvisited_query = """
            SELECT d.*
            FROM destinations d
            WHERE d.destination_id NOT IN (
                SELECT DISTINCT hotel.destination_id
                FROM bookings b
                JOIN hotel_bookings hb ON b.booking_id = hb.booking_id
                JOIN hotels hotel ON hb.hotel_id = hotel.hotel_id
                WHERE b.user_id = ? AND b.booking_status = 'completed'
                
                UNION
                
                SELECT DISTINCT tour.destination_id
                FROM bookings b
                JOIN tour_bookings tb ON b.booking_id = tb.booking_id
                JOIN tours tour ON tb.tour_id = tour.tour_id
                WHERE b.user_id = ? AND b.booking_status = 'completed'
            )
            """
            
            cursor = conn.cursor()
            cursor.execute(unvisited_query, (user_id, user_id))
            unvisited_destinations = []
            
            columns = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                unvisited_destinations.append(dict(zip(columns, row)))
                
            if not unvisited_destinations:
                print("User has visited all destinations")
                conn.close()
                return pd.DataFrame()
                
            # Convert to DataFrame
            unvisited_df = pd.DataFrame(unvisited_destinations)
            
            # Apply simple content-based filtering based on user preferences
            if 'preferred_climates' in user_features and user_features['preferred_climates']:
                climate_preferences = user_features['preferred_climates']
                climate_match = unvisited_df['climate_type'].isin(climate_preferences)
                
                # If we have matches based on climate, filter to those
                if climate_match.sum() > num_recommendations:
                    unvisited_df = unvisited_df[climate_match]
            
            # Get popular destinations based on ratings
            popular_query = """
            SELECT d.destination_id, 
                   AVG(CAST(r.rating AS FLOAT)) as avg_rating,
                   COUNT(DISTINCT r.review_id) as review_count,
                   COUNT(DISTINCT b.user_id) as booking_count
            FROM destinations d
            LEFT JOIN reviews r ON d.destination_id = r.entity_id AND r.entity_type = 'destination'
            LEFT JOIN hotels h ON d.destination_id = h.destination_id
            LEFT JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
            LEFT JOIN bookings b ON hb.booking_id = b.booking_id
            GROUP BY d.destination_id
            """
            
            popularity_df = pd.read_sql_query(popular_query, conn)
            
            # Merge with unvisited destinations
            recommendations = unvisited_df.merge(popularity_df, on='destination_id', how='left')
            
            # Fill NaN values
            recommendations['avg_rating'] = recommendations['avg_rating'].fillna(0)
            recommendations['review_count'] = recommendations['review_count'].fillna(0)
            recommendations['booking_count'] = recommendations['booking_count'].fillna(0)
            
            # Calculate a simple score based on popularity
            recommendations['score'] = (
                0.6 * recommendations['avg_rating'] + 
                0.2 * np.log1p(recommendations['review_count']) + 
                0.2 * np.log1p(recommendations['booking_count'])
            )
            
            # If there are no valid scores, add random scores
            if recommendations['score'].max() == 0:
                recommendations['score'] = [random.uniform(0.1, 0.9) for _ in range(len(recommendations))]
            
            # Normalize scores to 0-1 range
            max_score = recommendations['score'].max()
            if max_score > 0:
                recommendations['score'] = recommendations['score'] / max_score
            
            # Sort and return top recommendations
            recommendations = recommendations.sort_values('score', ascending=False).head(num_recommendations)
            conn.close()
            
            return recommendations
            
        except Exception as e:
            print(f"Error in fallback recommendations: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
            
    def get_destination_recommendations(self, user_id, num_recommendations=10):
        """
        Generate destination recommendations for a user.
        
        Args:
            user_id (int): The ID of the user to generate recommendations for
            num_recommendations (int): The number of recommendations to generate
            
        Returns:
            pd.DataFrame: DataFrame of recommended destinations
        """
        # First check if we have a trained model, if not, train it
        if not self.train_or_load_model():
            print("Failed to prepare recommendation model, using fallback")
            return self.fallback_recommendations(user_id, num_recommendations)
            
        try:
            print(f"Generating recommendations for user {user_id}")
            # Connect to database
            conn = self.connect_to_db()
            if conn is None:
                print("Failed to connect to database")
                return self.fallback_recommendations(user_id, num_recommendations)
                
            # Set the transaction isolation level to READ UNCOMMITTED to prevent locking issues
            cursor = conn.cursor()
            cursor.execute("SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
            
            # Get user features
            user_features = self.get_user_features(user_id)
            
            # Get destinations the user hasn't visited using a more reliable query with proper joins
            # This query avoids potential deadlocks by being more selective
            unvisited_query = """
            SELECT d.*
            FROM destinations d
            WHERE d.destination_id NOT IN (
                SELECT DISTINCT hotel.destination_id
                FROM bookings b
                JOIN hotel_bookings hb ON b.booking_id = hb.booking_id
                JOIN hotels hotel ON hb.hotel_id = hotel.hotel_id
                WHERE b.user_id = ? AND b.booking_status = 'completed'
                
                UNION
                
                SELECT DISTINCT tour.destination_id
                FROM bookings b
                JOIN tour_bookings tb ON b.booking_id = tb.booking_id
                JOIN tours tour ON tb.tour_id = tour.tour_id
                WHERE b.user_id = ? AND b.booking_status = 'completed'
            )
            """
            
            try:
                cursor = conn.cursor()
                cursor.execute(unvisited_query, (user_id, user_id))
                unvisited_destinations = []
                
                columns = [column[0] for column in cursor.description]
                for row in cursor.fetchall():
                    unvisited_destinations.append(dict(zip(columns, row)))
                    
                if not unvisited_destinations:
                    # If the user has visited all destinations, return empty DataFrame
                    print("User has visited all destinations")
                    conn.close()
                    return pd.DataFrame()
                    
                # Convert to DataFrame
                unvisited_df = pd.DataFrame(unvisited_destinations)
                
                # If model training failed, use a simple recommendation approach
                if self.model is None:
                    # Use a random selection of destinations as a fallback
                    print("Using fallback recommendation method (model is None)")
                    recommendations = unvisited_df.sample(min(num_recommendations, len(unvisited_df)))
                    recommendations['score'] = [random.uniform(0.5, 0.9) for _ in range(len(recommendations))]
                    conn.close()
                    return recommendations.sort_values('score', ascending=False)
                
                # Prepare features for prediction
                X_test = self.prepare_features_for_prediction(unvisited_df, user_features)
                
                if X_test.empty:
                    print("No valid features for prediction")
                    # Fall back to content-based recommendations
                    return self.fallback_recommendations(user_id, num_recommendations)
                
                # Make predictions
                try:
                    scores = self.model.predict_proba(X_test)[:, 1]  # Get probability of class 1
                except Exception as e:
                    print(f"Error during model prediction: {str(e)}")
                    traceback.print_exc()
                    # Fall back to content-based recommendations
                    conn.close()
                    return self.fallback_recommendations(user_id, num_recommendations)
                
                # Create recommendations DataFrame with all destination details
                recommendations = unvisited_df.copy()
                recommendations['score'] = scores
                
                # Ensure we have all required columns for map display
                required_cols = ['destination_id', 'name', 'country', 'latitude', 'longitude']
                missing_cols = [col for col in required_cols if col not in recommendations.columns]
                
                if missing_cols:
                    print(f"Warning: Missing columns in recommendations: {missing_cols}")
                    # Try to get any missing columns from the database
                    if 'destination_id' in recommendations.columns:
                        destination_ids = recommendations['destination_id'].tolist()
                        placeholders = ','.join(['?' for _ in destination_ids])
                        
                        # Use a single query with proper transaction isolation to get complete details
                        complete_dest_query = f"""
                        SELECT destination_id, name, country, region, latitude, longitude, 
                               climate_type, best_season_to_visit, popularity_score, average_rating
                        FROM destinations WITH (NOLOCK)
                        WHERE destination_id IN ({placeholders})
                        """
                        
                        cursor.execute(complete_dest_query, destination_ids)
                        complete_destinations = []
                        
                        columns = [column[0] for column in cursor.description]
                        for row in cursor.fetchall():
                            complete_destinations.append(dict(zip(columns, row)))
                        
                        if complete_destinations:
                            # Merge with original recommendations to preserve scores
                            complete_df = pd.DataFrame(complete_destinations)
                            recommendations = pd.merge(
                                recommendations,
                                complete_df,
                                on='destination_id',
                                how='left',
                                suffixes=('', '_complete')
                            )
                            
                            # Use complete columns where original is missing
                            for col in missing_cols:
                                if f"{col}_complete" in recommendations.columns:
                                    recommendations[col] = recommendations[f"{col}_complete"]
                                    
                            # Remove the duplicate columns with _complete suffix
                            recommendations = recommendations[[c for c in recommendations.columns if not c.endswith('_complete')]]
                
                # Sort and return top recommendations
                recommendations = recommendations.sort_values('score', ascending=False).head(num_recommendations)
                conn.close()
                return recommendations
                
            except Exception as e:
                print(f"Error retrieving unvisited destinations: {str(e)}")
                traceback.print_exc()
                conn.close()
                return pd.DataFrame()
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def recommend_for_all_users(self, sample_size=None):
        """
        Generate recommendations for all users.
        
        Args:
            sample_size: Optional number of users to sample (for demo purposes)
            
        Returns:
            Dictionary with user IDs as keys and recommendation DataFrames as values
        """
        conn = self.connect_to_db()
        
        # Get all user IDs
        users_df = pd.read_sql_query("SELECT user_id FROM users", conn)
        user_ids = users_df['user_id'].tolist()
        
        conn.close()
        
        # Optionally sample users
        if sample_size is not None and sample_size < len(user_ids):
            user_ids = np.random.choice(user_ids, sample_size, replace=False)
        
        # Generate recommendations for each user
        recommendations = {}
        for user_id in user_ids:
            print(f"Generating recommendations for user {user_id}...")
            user_recs = self.get_destination_recommendations(user_id)
            recommendations[user_id] = user_recs
        
        return recommendations

if __name__ == "__main__":
    print("Initializing travel recommendation system...")
    
    # Initialize the recommendation system
    rec_system = TravelRecommendationSystem()
    
    # Train the model
    rec_system.train_model()
    
    # Test recommendations for a specific user
    user_id = 1  # Example user ID
    print(f"\nGenerating recommendations for user {user_id}...")
    recommendations = rec_system.get_destination_recommendations(user_id)
    
    if not recommendations.empty:
        print("\nTop recommended destinations:")
        print(recommendations[['name', 'country', 'region', 'score']].head(5))
    else:
        print("No recommendations generated.") 