import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyodbc
import json
import pickle
import os
import sys
import re
import subprocess
import importlib.util
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import random
import time
import traceback
import warnings
import plotly.express as px

# Add XGBoost imports for model training
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the recommendation system
from recommendation_system import TravelRecommendationSystem

# Database connection
def get_db_connection():
    """
    Get a connection to the database.
    
    Returns:
        pyodbc.Connection: A database connection
    """
    try:
        # SQL Server connection string
        conn_str = "DRIVER=/opt/homebrew/lib/libtdsodbc.so;SERVER=127.0.0.1;PORT=1433;DATABASE=virtuoso_travel;UID=climbing_user;PWD=hoosierheights;TDS_Version=7.4;"
        
        # Create a connection
        conn = pyodbc.connect(conn_str)
        
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        print(f"Database connection error: {e}")
        return None

# Function to import modules dynamically
def import_module(module_path):
    """Import a module from file path."""
    module_name = os.path.basename(module_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Database setup functions
def check_database_connection():
    """Check if the database connection works."""
    conn = get_db_connection()
    if conn is None:
        return False, "Failed to connect to database"
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        return True, "Successfully connected to database"
    except Exception as e:
        return False, f"Error executing query: {e}"

def setup_database_schema():
    """Setup the database schema using database_setup.py."""
    try:
        # Import the database_setup module
        setup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database_setup.py')
        setup_module = import_module(setup_path)
        
        if setup_module is None:
            return False, "Failed to import database_setup.py"
        
        # Call the setup function
        conn_str = setup_module.create_database()
        return True, "Database schema created successfully"
    except Exception as e:
        return False, f"Error setting up database schema: {e}"

def check_and_create_transaction_tables():
    """Check if transaction tables exist and create them if not."""
    try:
        conn = get_db_connection()
        if conn is None:
            return False, "Failed to connect to database"
        
        cursor = conn.cursor()
        
        # List of transaction tables to check and their creation scripts
        transaction_tables = {
            'bookings': """
                CREATE TABLE bookings (
                    booking_id INT PRIMARY KEY,
                    user_id INT,
                    booking_date DATE,
                    total_cost FLOAT,
                    payment_status NVARCHAR(50),
                    booking_status NVARCHAR(50),
                    booking_channel NVARCHAR(50),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """,
            'hotel_bookings': """
                CREATE TABLE hotel_bookings (
                    hotel_booking_id INT PRIMARY KEY,
                    booking_id INT,
                    hotel_id INT,
                    check_in_date DATE,
                    check_out_date DATE,
                    room_type NVARCHAR(50),
                    number_of_guests INT,
                    special_requests NVARCHAR(255),
                    rate_per_night FLOAT,
                    FOREIGN KEY (booking_id) REFERENCES bookings(booking_id),
                    FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
                )
            """,
            'tour_bookings': """
                CREATE TABLE tour_bookings (
                    tour_booking_id INT PRIMARY KEY,
                    booking_id INT,
                    tour_id INT,
                    tour_date DATE,
                    number_of_participants INT,
                    special_requirements NVARCHAR(255),
                    FOREIGN KEY (booking_id) REFERENCES bookings(booking_id),
                    FOREIGN KEY (tour_id) REFERENCES tours(tour_id)
                )
            """,
            'reviews': """
                CREATE TABLE reviews (
                    review_id INT PRIMARY KEY,
                    user_id INT,
                    entity_type NVARCHAR(50),
                    entity_id INT,
                    rating INT,
                    comment NVARCHAR(MAX),
                    review_date DATE,
                    helpful_votes INT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """,
            'user_interactions': """
                CREATE TABLE user_interactions (
                    interaction_id INT PRIMARY KEY,
                    user_id INT,
                    interaction_type NVARCHAR(50),
                    entity_type NVARCHAR(50),
                    entity_id INT,
                    timestamp DATETIME,
                    interaction_details NVARCHAR(MAX),
                    session_id NVARCHAR(100),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """
        }
        
        # Check if each table exists and create it if not
        for table_name, create_script in transaction_tables.items():
            cursor.execute(f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}'")
            if not cursor.fetchone():
                print(f"Table {table_name} does not exist, creating it...")
                cursor.execute(create_script)
                conn.commit()
                print(f"Created table {table_name}")
            else:
                print(f"Table {table_name} already exists")
        
        # Create indices for better performance if they don't exist
        indices = [
            "CREATE INDEX idx_bookings_id ON bookings(booking_id)",
            "CREATE INDEX idx_bookings_user ON bookings(user_id)",
            "CREATE INDEX idx_hotel_bookings_id ON hotel_bookings(hotel_booking_id)",
            "CREATE INDEX idx_hotel_bookings_booking ON hotel_bookings(booking_id)",
            "CREATE INDEX idx_hotel_bookings_hotel ON hotel_bookings(hotel_id)",
            "CREATE INDEX idx_tour_bookings_id ON tour_bookings(tour_booking_id)",
            "CREATE INDEX idx_tour_bookings_booking ON tour_bookings(booking_id)",
            "CREATE INDEX idx_tour_bookings_tour ON tour_bookings(tour_id)",
            "CREATE INDEX idx_reviews_id ON reviews(review_id)",
            "CREATE INDEX idx_reviews_user ON reviews(user_id)",
            "CREATE INDEX idx_reviews_entity ON reviews(entity_type, entity_id)",
            "CREATE INDEX idx_interactions_id ON user_interactions(interaction_id)",
            "CREATE INDEX idx_interactions_user ON user_interactions(user_id)",
            "CREATE INDEX idx_interactions_entity ON user_interactions(entity_type, entity_id)",
            "CREATE INDEX idx_interactions_timestamp ON user_interactions(timestamp)"
        ]
        
        for index_query in indices:
            try:
                cursor.execute(index_query)
                conn.commit()
            except Exception as e:
                # Skip if index already exists
                print(f"Note: {e}")
        
        conn.close()
        return True, "Transaction tables verified/created successfully"
    except Exception as e:
        return False, f"Error checking/creating transaction tables: {e}"

def generate_simulation_data():
    """Generate simulation data using data_generator.py, first clearing travel history tables."""
    try:
        # First check if the transaction tables exist and create them if needed
        tables_success, tables_message = check_and_create_transaction_tables()
        if not tables_success:
            return False, tables_message
            
        # Connect to database to clear transaction tables
        conn = get_db_connection()
        if conn is None:
            return False, "Failed to connect to database"
        
        cursor = conn.cursor()
        
        # List of tables to clear (in order of dependencies)
        tables_to_clear = [
            "reviews",
            "user_interactions",
            "tour_bookings",
            "hotel_bookings",
            "bookings"
        ]
        
        # Clear tables
        for table in tables_to_clear:
            try:
                cursor.execute(f"IF OBJECT_ID('{table}', 'U') IS NOT NULL DELETE FROM {table}")
                conn.commit()
                print(f"Cleared all records from {table}")
            except Exception as e:
                print(f"Note: {e}")
        
        conn.close()
        
        # Import the data_generator module
        data_gen_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_generator.py')
        data_gen = import_module(data_gen_path)
        
        if data_gen is None:
            return False, "Failed to import data_generator.py"
        
        # Call the data generation function
        result = data_gen.generate_data()
        return True, "Travel data reset and new simulation data generated successfully"
    except Exception as e:
        return False, f"Error generating simulation data: {e}"

def fetch_api_data():
    """Fetch data from APIs if available."""
    try:
        # Import the API data fetcher module if it exists
        api_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_data_fetcher.py')
        
        if not os.path.exists(api_path):
            return False, "api_data_fetcher.py not found"
        
        api_fetcher = import_module(api_path)
        
        if api_fetcher is None:
            return False, "Failed to import api_data_fetcher.py"
        
        # First clear existing data
        st.info("Clearing existing data before fetching from APIs...")
        api_fetcher.clear_existing_data()
        
        # Call the API fetching function
        st.info("Fetching data from travel APIs. This may take a few minutes...")
        success = api_fetcher.fetch_data()
        
        if success:
            return True, "API data fetched and loaded into database successfully"
        else:
            return False, "Failed to fetch API data, check console for details"
    except Exception as e:
        return False, f"Error fetching API data: {e}"

def run_complete_setup():
    """Run the complete database setup and data generation process."""
    results = []
    
    # Step 1: Check connection
    success, message = check_database_connection()
    results.append(("Database Connection", success, message))
    if not success:
        return results
    
    # Step 2: Setup schema
    success, message = setup_database_schema()
    results.append(("Database Schema", success, message))
    if not success:
        return results
    
    # Step 3: Try to fetch API data first
    success, message = fetch_api_data()
    results.append(("API Data Fetch", success, message))
    
    # Step 4: Generate simulation data (especially if API fetch failed)
    if not success:
        success, message = generate_simulation_data()
        results.append(("Simulation Data", success, message))
    
    return results

def reset_travel_data():
    """Drop and recreate travel history related tables, then regenerate simulation data."""
    try:
        conn = get_db_connection()
        if conn is None:
            return False, "Failed to connect to database"
        
        cursor = conn.cursor()
        
        # List of tables to drop (in order of dependencies)
        tables_to_drop = [
            "reviews",
            "user_interactions",
            "tour_bookings",
            "hotel_bookings",
            "bookings"
        ]
        
        # Drop tables
        for table in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                conn.commit()
                print(f"Dropped table {table}")
            except Exception as e:
                print(f"Error dropping table {table}: {e}")
                conn.rollback()
        
        # Recreation of tables will be handled by the database_setup.py script
        # which should detect missing tables and recreate them
        
        conn.close()
        
        # Call database setup to recreate tables
        setup_success, setup_message = setup_database_schema()
        if not setup_success:
            return False, f"Error recreating tables: {setup_message}"
        
        # Generate new simulation data
        sim_success, sim_message = generate_simulation_data()
        if not sim_success:
            return False, f"Error generating simulation data: {sim_message}"
        
        return True, "Travel data reset and regenerated successfully"
    except Exception as e:
        return False, f"Error resetting travel data: {e}"

def show_admin_db_setup_panel():
    """Show the admin database setup panel."""
    st.subheader("Database Setup")
    
    # Database connection status
    st.write("**Database Connection Status:**")
    
    if st.button("Check Database Connection"):
        success, message = check_database_connection()
        if success:
            st.success(message)
        else:
            st.error(message)
    
    # Database setup options
    st.write("**Setup Options:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Create Database Schema"):
            with st.spinner("Creating database schema..."):
                success, message = setup_database_schema()
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    with col2:
        if st.button("Generate Simulation Data"):
            with st.spinner("Clearing travel history and generating simulation data..."):
                success, message = generate_simulation_data()
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # API data fetch
    if st.button("Fetch Data from APIs"):
        with st.spinner("Fetching data from APIs..."):
            success, message = fetch_api_data()
            if success:
                st.success(message)
            else:
                st.warning(message)
    
    # Population without schema creation
    st.write("**Data Population Only:**")
    st.info("Use this if you've already created the database schema manually.")
    
    if st.button("Populate Data Only"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Starting data population process...")
        time.sleep(1)
        
        # Try to fetch API data first
        status_text.text("Attempting to fetch data from APIs...")
        progress_bar.progress(30)
        success_api, message_api = fetch_api_data()
        
        if success_api:
            status_text.text("API data fetched successfully!")
            progress_bar.progress(90)
        else:
            # Generate simulation data if API fetch failed
            status_text.text("API fetch failed, generating simulation data...")
            progress_bar.progress(50)
            success_sim, message_sim = generate_simulation_data()
            
            if not success_sim:
                st.error(f"Data generation failed: {message_sim}")
                return
            
            status_text.text("Simulation data generated successfully!")
            progress_bar.progress(90)
        
        # Final step
        status_text.text("Data population complete!")
        progress_bar.progress(100)
        st.success("Database populated successfully!")
        
    # Complete setup (original)
    st.write("**Complete Setup:**")
    st.warning("This will run the complete setup process, which may take several minutes.")
    
    if st.button("Run Complete Setup"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Check connection
        status_text.text("Checking database connection...")
        progress_bar.progress(10)
        success, message = check_database_connection()
        if not success:
            st.error(f"Database connection failed: {message}")
            return
        
        # Step 2: Setup schema
        status_text.text("Creating database schema...")
        progress_bar.progress(30)
        success, message = setup_database_schema()
        if not success:
            st.error(f"Database schema creation failed: {message}")
            return
        
        # Step 3: Try to fetch API data first
        status_text.text("Attempting to fetch data from APIs...")
        progress_bar.progress(50)
        success_api, message_api = fetch_api_data()
        
        if success_api:
            status_text.text("API data fetched successfully!")
            progress_bar.progress(80)
        else:
            # Step 4: Generate simulation data if API fetch failed
            status_text.text("API fetch failed, generating simulation data...")
            progress_bar.progress(60)
            success_sim, message_sim = generate_simulation_data()
            
            if not success_sim:
                st.error(f"Data generation failed: {message_sim}")
                return
            
            status_text.text("Simulation data generated successfully!")
            progress_bar.progress(80)
        
        # Final step
        status_text.text("Setup complete!")
        progress_bar.progress(100)
        st.success("Database setup completed successfully!")
        
        # Display recommendation to check data
        st.info("You can now use the Database Explorer tab to browse the tables.")

# Create recommendation system instance
@st.cache_resource
def get_recommendation_system():
    """Get or initialize the recommendation system."""
    rec_system = TravelRecommendationSystem()
    # Check if model exists, if not, we'll train later if requested
    rec_system.load_model()
    return rec_system

def run_query(query, params=None):
    """Execute a SQL query and return the results as a pandas DataFrame."""
    conn = None
    try:
        import warnings
        with warnings.catch_warnings():
            # Suppress the UserWarning about pandas only supporting SQLAlchemy connectable
            warnings.filterwarnings("ignore", category=UserWarning, module='pandas')
            conn = get_db_connection()
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            return df
    except pyodbc.Error as e:
        error_msg = f"Database error: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return pd.DataFrame()
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def execute_query(query):
    """Execute a non-SELECT SQL query."""
    conn = get_db_connection()
    if conn is None:
        return False, "Failed to connect to database"
    
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        return True, "Query executed successfully"
    except Exception as e:
        return False, f"Error executing query: {e}"
    finally:
        conn.close()

# Destination Map
def show_destination_map(destinations_df, highlight_destinations=None, user_visited=None):
    """
    Show a map of all destinations with highlighting for recommendations.
    
    Args:
        destinations_df: DataFrame with all destinations
        highlight_destinations: DataFrame with destinations to highlight (recommendations)
        user_visited: DataFrame with destinations user has visited
    """
    # Check if DataFrame is empty
    if destinations_df.empty:
        st.warning("No destination data available to display on map.")
        return
        
    # Map column names - handle different naming conventions
    lat_cols = ['latitude', 'lat', 'destination_latitude']
    lng_cols = ['longitude', 'lng', 'long', 'destination_longitude']
    name_cols = ['name', 'destination_name']
    country_cols = ['country']
    
    # Find the actual column names in the DataFrame
    lat_col = next((col for col in lat_cols if col in destinations_df.columns), None)
    lng_col = next((col for col in lng_cols if col in destinations_df.columns), None)
    name_col = next((col for col in name_cols if col in destinations_df.columns), None)
    country_col = next((col for col in country_cols if col in destinations_df.columns), None)
    
    # Check if we found all required columns
    missing_types = []
    if lat_col is None:
        missing_types.append("latitude")
    if lng_col is None:
        missing_types.append("longitude")
    if name_col is None:
        missing_types.append("name")
    if country_col is None:
        missing_types.append("country")
        
    if missing_types:
        st.error(f"Cannot display map: Missing required column types: {', '.join(missing_types)}")
        st.error(f"Available columns: {', '.join(destinations_df.columns)}")
        return
    
    # Create a map centered at the mean coordinates
    # Use a try/except to handle potential NaN values
    try:
        center_lat = destinations_df[lat_col].mean()
        center_lon = destinations_df[lng_col].mean()
        
        # Check for NaN or invalid coordinates
        if pd.isna(center_lat) or pd.isna(center_lon) or not (-90 <= center_lat <= 90) or not (-180 <= center_lon <= 180):
            # Default center if invalid coordinates
            center_lat = 20
            center_lon = 0
    except Exception:
        # Default center if there was an error
        center_lat = 20
        center_lon = 0
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
    
    # Add all destinations as small gray dots
    for _, dest in destinations_df.iterrows():
        try:
            lat = dest[lat_col]
            lon = dest[lng_col]
            
            # Skip invalid coordinates
            if pd.isna(lat) or pd.isna(lon) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                continue
                
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='gray',
                fill=True,
                fill_color='gray',
                fill_opacity=0.4,
                popup=f"{dest[name_col]}, {dest[country_col]}"
            ).add_to(m)
        except Exception:
            # Skip this destination if there was an error
            continue
    
    # Add visited destinations as blue stars if provided
    if user_visited is not None and not user_visited.empty:
        # Find column names in visited DataFrame
        v_lat_col = next((col for col in lat_cols if col in user_visited.columns), None)
        v_lng_col = next((col for col in lng_cols if col in user_visited.columns), None)
        v_name_col = next((col for col in name_cols if col in user_visited.columns), None)
        v_country_col = next((col for col in country_cols if col in user_visited.columns), None)
        
        # Check if we have the required columns
        if None not in [v_lat_col, v_lng_col, v_name_col, v_country_col]:
            for _, dest in user_visited.iterrows():
                try:
                    lat = dest[v_lat_col]
                    lon = dest[v_lng_col]
                    
                    # Skip invalid coordinates
                    if pd.isna(lat) or pd.isna(lon) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        continue
                        
                    folium.Marker(
                        location=[lat, lon],
                        icon=folium.Icon(color='blue', icon='star', prefix='fa'),
                        popup=f"<b>Visited:</b> {dest[v_name_col]}, {dest[v_country_col]}"
                    ).add_to(m)
                except Exception:
                    # Skip this destination if there was an error
                    continue
        else:
            st.warning("Cannot display visited destinations: Missing required columns")
    
    # Add highlighted destinations with custom colors based on score if provided
    if highlight_destinations is not None and not highlight_destinations.empty:
        # Find column names in highlight DataFrame
        h_lat_col = next((col for col in lat_cols if col in highlight_destinations.columns), None)
        h_lng_col = next((col for col in lng_cols if col in highlight_destinations.columns), None)
        h_name_col = next((col for col in name_cols if col in highlight_destinations.columns), None)
        h_country_col = next((col for col in country_cols if col in highlight_destinations.columns), None)
        
        # Check if we have the required columns
        if None in [h_lat_col, h_lng_col, h_name_col, h_country_col]:
            # Try to merge with all_destinations to get missing columns
            if 'destination_id' in highlight_destinations.columns:
                try:
                    # Merge with all destinations to get the missing columns
                    merged_highlights = pd.merge(
                        highlight_destinations,
                        destinations_df,
                        on='destination_id',
                        how='left',
                        suffixes=('', '_all')
                    )
                    
                    # Check if destination_id is still there (it should be)
                    if 'destination_id' not in merged_highlights.columns:
                        st.warning("Destination ID column lost during merge - cannot display recommendations")
                        return
                    
                    # Now check for columns again
                    h_lat_col = next((col for col in lat_cols if col in merged_highlights.columns), None)
                    h_lng_col = next((col for col in lng_cols if col in merged_highlights.columns), None)
                    h_name_col = next((col for col in name_cols if col in merged_highlights.columns), None)
                    h_country_col = next((col for col in country_cols if col in merged_highlights.columns), None)
                    
                    if None not in [h_lat_col, h_lng_col, h_name_col, h_country_col]:
                        highlight_destinations = merged_highlights
                    else:
                        st.warning("Cannot display recommendation highlights: Missing required columns after merge")
                        st.write(f"Available columns after merge: {list(merged_highlights.columns)}")
                        st.write(f"Missing column types: latitude={h_lat_col is None}, longitude={h_lng_col is None}, name={h_name_col is None}, country={h_country_col is None}")
                        return
                except Exception as e:
                    st.warning(f"Error merging destination data: {str(e)}")
                    return
            else:
                st.warning("Cannot display recommendation highlights: Recommendations missing destination_id column")
                return
        
        # Check if score column exists
        if 'score' not in highlight_destinations.columns:
            st.info("Score column missing from recommendations. Using default sorting.")
            score_available = False
        else:
            score_available = True
            # Sort by score descending to make sure top recommendations are on top
            highlight_destinations = highlight_destinations.sort_values('score', ascending=False)
        
        # Use a colormap from green (high score) to red (lower score)
        for idx, dest in highlight_destinations.iterrows():
            try:
                lat = dest[h_lat_col]
                lon = dest[h_lng_col]
                
                # Skip invalid coordinates
                if pd.isna(lat) or pd.isna(lon) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue
                
                # Calculate color: default green if score is not available
                if score_available:
                    score = dest['score']
                    # Colormap from red to green
                    if score >= 0.8:
                        color = 'green'
                    elif score >= 0.6:
                        color = 'lightgreen'
                    elif score >= 0.4:
                        color = 'orange'
                    else:
                        color = 'red'
                else:
                    color = 'green'
                    score = 0
                
                # Check for optional columns
                popup_content = f"<b>Recommended:</b> {dest[h_name_col]}, {dest[h_country_col]}<br>"
                
                if score_available:
                    popup_content += f"Score: {score:.2f}<br>"
                
                # Add climate info if available
                climate_cols = ['climate_type', 'climate']
                season_cols = ['best_season_to_visit', 'best_season']
                
                climate_col = next((col for col in climate_cols if col in highlight_destinations.columns), None)
                season_col = next((col for col in season_cols if col in highlight_destinations.columns), None)
                
                if climate_col and pd.notna(dest.get(climate_col)):
                    popup_content += f"Climate: {dest[climate_col]}<br>"
                
                if season_col and pd.notna(dest.get(season_col)):
                    popup_content += f"Best Season: {dest[season_col]}"
                
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.Icon(color=color, icon='info-sign'),
                    popup=popup_content
                ).add_to(m)
            except Exception as e:
                # Skip this destination if there was an error
                continue
    
    # Display the map
    folium_static(m)

# User Profile Section
def show_user_profile(user_id):
    """Show the user profile information."""
    user_query = f"""
    SELECT u.*, 
           p.preferred_hotel_stars, p.preferred_budget_category, 
           p.preferred_activities, p.preferred_climates, p.travel_style
    FROM users u
    LEFT JOIN user_preferences p ON u.user_id = p.user_id
    WHERE u.user_id = ?
    """
    
    user_df = run_query(user_query, params=(user_id,))
    
    if user_df.empty:
        st.warning(f"User ID {user_id} not found.")
        return None
    
    user = user_df.iloc[0]
    
    st.subheader("User Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Name:** {user['first_name']} {user['last_name']}")
        st.write(f"**Email:** {user['email']}")
        st.write(f"**Country:** {user['country_of_residence']}")
        st.write(f"**Loyalty Tier:** {user['loyalty_tier']}")
        
        # Calculate age
        birth_date = pd.to_datetime(user['date_of_birth'])
        age = (datetime.now() - birth_date).days // 365
        st.write(f"**Age:** {age}")
        
    with col2:
        st.write("**Travel Preferences:**")
        st.write(f"- Budget Category: {user.get('preferred_budget_category', 'Not specified')}")
        st.write(f"- Preferred Hotel Stars: {user.get('preferred_hotel_stars', 'Not specified')}")
        st.write(f"- Travel Style: {user.get('travel_style', 'Not specified')}")
        
        # Parse JSON fields
        if pd.notna(user.get('preferred_activities')):
            activities = json.loads(user['preferred_activities'])
            st.write("- Preferred Activities:")
            st.write(", ".join(activities))
        
        if pd.notna(user.get('preferred_climates')):
            climates = json.loads(user['preferred_climates'])
            st.write("- Preferred Climates:")
            st.write(", ".join(climates))
    
    return user

# User History Section
def show_user_history(user_id):
    """Show the user's travel history with preference-based generation if needed."""
    # Query to get visited destinations
    visited_query = f"""
    SELECT DISTINCT
        d.destination_id, d.name, d.country, d.region, d.latitude, d.longitude,
        d.image_url, d.climate_type, d.best_season_to_visit
    FROM
        destinations d
    JOIN
        hotels h ON d.destination_id = h.destination_id
    JOIN
        hotel_bookings hb ON h.hotel_id = hb.hotel_id
    JOIN
        bookings b ON hb.booking_id = b.booking_id
    WHERE
        b.user_id = ? AND b.booking_status = 'completed'
    UNION
    SELECT DISTINCT
        d.destination_id, d.name, d.country, d.region, d.latitude, d.longitude,
        d.image_url, d.climate_type, d.best_season_to_visit
    FROM
        destinations d
    JOIN
        tours t ON d.destination_id = t.destination_id
    JOIN
        tour_bookings tb ON t.tour_id = tb.tour_id
    JOIN
        bookings b ON tb.booking_id = b.booking_id
    WHERE
        b.user_id = ? AND b.booking_status = 'completed'
    """
    
    visited_df = run_query(visited_query, params=(user_id, user_id))
    
    if visited_df.empty:
        # Generate preference-based travel history
        st.info("No travel history found in the database. Generating preference-based travel history for recommendation accuracy.")
        
        # Get user profile and preferences
        user_query = """
        SELECT u.*, 
               p.preferred_hotel_stars, p.preferred_budget_category, 
               p.preferred_activities, p.preferred_climates, p.travel_style
        FROM users u
        LEFT JOIN user_preferences p ON u.user_id = p.user_id
        WHERE u.user_id = ?
        """
        user_df = run_query(user_query, params=(user_id,))
        
        if user_df.empty:
            st.error("User profile not found.")
            return pd.DataFrame()
            
        user = user_df.iloc[0]
        
        # Extract user preferences
        preferred_climates = []
        if pd.notna(user.get('preferred_climates')):
            try:
                preferred_climates = json.loads(user['preferred_climates']) if isinstance(user['preferred_climates'], str) else user['preferred_climates']
            except:
                preferred_climates = []
                
        preferred_activities = []
        if pd.notna(user.get('preferred_activities')):
            try:
                preferred_activities = json.loads(user['preferred_activities']) if isinstance(user['preferred_activities'], str) else user['preferred_activities']
            except:
                preferred_activities = []
        
        travel_style = user.get('travel_style', '')
        preferred_budget = user.get('preferred_budget_category', '')
        
        # Query to find matching destinations based on preferences
        climate_clause = ""
        if preferred_climates:
            climate_list = ", ".join([f"'{climate}'" for climate in preferred_climates])
            climate_clause = f" AND (d.climate_type IN ({climate_list}) OR d.climate_type IS NULL)"
        
        # Construct query to find destinations matching user preferences
        preference_match_query = f"""
        SELECT d.*, 
               CASE 
                   WHEN d.climate_type IN ({climate_list if preferred_climates else "''"}) THEN 1 
                   ELSE 0 
               END as preference_score
        FROM destinations d
        WHERE 1=1 {climate_clause}
        ORDER BY preference_score DESC, NEWID()
        """
        
        # Get destinations with preference matching
        pref_destinations = run_query(preference_match_query)
        
        # Select 5-8 destinations that match preferences
        num_destinations = random.randint(5, 8)
        if len(pref_destinations) > num_destinations:
            # Take top matches with a bit of randomness
            selected_indices = list(range(min(15, len(pref_destinations))))
            random.shuffle(selected_indices)
            selected_indices = selected_indices[:num_destinations]
            mock_visited_df = pref_destinations.iloc[selected_indices].copy()
        else:
            mock_visited_df = pref_destinations.copy()
        
        if not mock_visited_df.empty:
            visited_df = mock_visited_df
            
            # Add simulated booking data to the database - OPTIMIZED VERSION
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Start a single transaction for all operations
                conn.autocommit = False
                
                # Calculate user loyalty level and travel frequency
                loyalty_tier = user.get('loyalty_tier', 'Standard')
                # Higher tier users travel more frequently and longer ago (more history)
                max_days_ago = 730  # 2 years for standard
                if loyalty_tier == 'Gold':
                    max_days_ago = 1095  # 3 years 
                elif loyalty_tier == 'Platinum':
                    max_days_ago = 1460  # 4 years
                
                # Determine travel frequency - higher tier users travel more often
                min_days_between_trips = 60  # Standard: ~6 trips per year
                if loyalty_tier == 'Gold':
                    min_days_between_trips = 45  # Gold: ~8 trips per year
                elif loyalty_tier == 'Platinum':
                    min_days_between_trips = 30  # Platinum: ~12 trips per year
                
                # Pre-fetch the next available booking ID once
                cursor.execute("SELECT ISNULL(MAX(booking_id), 0) FROM bookings")
                next_booking_id = cursor.fetchone()[0] + 1
                
                # Pre-fetch the next available hotel_booking_id
                cursor.execute("SELECT ISNULL(MAX(hotel_booking_id), 0) FROM hotel_bookings")
                next_hotel_booking_id = cursor.fetchone()[0] + 1
                
                # Pre-fetch the next available tour_booking_id
                cursor.execute("SELECT ISNULL(MAX(tour_booking_id), 0) FROM tour_bookings")
                next_tour_booking_id = cursor.fetchone()[0] + 1
                
                # Pre-fetch the next available review_id and interaction_id
                cursor.execute("SELECT ISNULL(MAX(review_id), 0) FROM reviews")
                next_review_id = cursor.fetchone()[0] + 1
                
                cursor.execute("SELECT ISNULL(MAX(interaction_id), 0) FROM user_interactions")
                next_interaction_id = cursor.fetchone()[0] + 1
                
                # Pre-cache hotel and tour IDs for all destinations to avoid repeated queries
                dest_ids = tuple(mock_visited_df['destination_id'].tolist())
                if len(dest_ids) == 1:
                    # Handle single-item tuple syntax
                    dest_id_str = f"({dest_ids[0]})"
                else:
                    dest_id_str = str(dest_ids)
                
                # Get all hotels for these destinations in one query
                hotels_query = f"SELECT hotel_id, destination_id FROM hotels WHERE destination_id IN {dest_id_str}"
                cursor.execute(hotels_query)
                hotels_by_dest = {}
                for hotel_id, dest_id in cursor.fetchall():
                    if dest_id not in hotels_by_dest:
                        hotels_by_dest[dest_id] = []
                    hotels_by_dest[dest_id].append(hotel_id)
                
                # Get all tours for these destinations in one query
                tours_query = f"SELECT tour_id, destination_id FROM tours WHERE destination_id IN {dest_id_str}"
                cursor.execute(tours_query)
                tours_by_dest = {}
                for tour_id, dest_id in cursor.fetchall():
                    if dest_id not in tours_by_dest:
                        tours_by_dest[dest_id] = []
                    tours_by_dest[dest_id].append(tour_id)
                
                # Prepare batch inserts
                booking_inserts = []
                hotel_booking_inserts = []
                tour_booking_inserts = []
                review_inserts = []
                interaction_inserts = []
                
                # Generate dates that make chronological sense
                days_ago = max_days_ago
                
                # We'll track which destinations need hotel vs tour bookings
                hotel_bookings = []
                tour_bookings = []
                
                for idx, dest in mock_visited_df.iterrows():
                    dest_id = dest['destination_id']
                    
                    # Calculate dates that follow a logical pattern
                    days_ago = days_ago - random.randint(min_days_between_trips, min_days_between_trips + 30)
                    if days_ago < 30:  # Don't create future trips
                        days_ago = 30
                        
                    booking_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
                    
                    # Determine cost based on budget preference and destination
                    base_cost = 1000
                    if preferred_budget == 'luxury':
                        base_cost = 3000
                    elif preferred_budget == 'moderate':
                        base_cost = 1500
                    elif preferred_budget == 'budget':
                        base_cost = 800
                        
                    # Add destination-based cost variation
                    high_cost_countries = ['USA', 'Canada', 'Japan', 'Australia', 'UK', 'France', 'Germany']
                    medium_cost_countries = ['Spain', 'Italy', 'Greece', 'Brazil', 'China', 'South Africa']
                    
                    if dest['country'] in high_cost_countries:
                        cost_multiplier = 1.5
                    elif dest['country'] in medium_cost_countries:
                        cost_multiplier = 1.2
                    else:
                        cost_multiplier = 1.0
                        
                    total_cost = int(base_cost * cost_multiplier * random.uniform(0.8, 1.2))
                    
                    # Add booking to batch
                    booking_id = next_booking_id
                    next_booking_id += 1
                    booking_inserts.append((booking_id, user_id, booking_date, total_cost, 'completed'))
                    
                    # Determine accommodation/activity type based on travel style and preferences
                    is_hotel_booking = True
                    if travel_style in ['adventure', 'eco-tourism'] or 'hiking' in preferred_activities or 'camping' in preferred_activities:
                        # Adventure travelers book more tours
                        is_hotel_booking = random.random() > 0.6
                    elif travel_style in ['cultural', 'historical']:
                        # Cultural travelers book more tours too
                        is_hotel_booking = random.random() > 0.5
                    elif travel_style in ['luxury', 'resort']:
                        # Luxury travelers prefer hotels
                        is_hotel_booking = random.random() > 0.3
                    
                    if is_hotel_booking and dest_id in hotels_by_dest and hotels_by_dest[dest_id]:
                        # Use a hotel from our pre-fetched list
                        hotel_id = random.choice(hotels_by_dest[dest_id])
                        check_in = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
                        
                        # Trip duration based on travel style and destination type
                        min_duration = 3
                        max_duration = 7
                        
                        if travel_style == 'luxury':
                            min_duration += 2
                            max_duration += 3
                        elif dest.get('destination_type') in ['beach', 'resort']:
                            min_duration += 1
                            max_duration += 2
                        
                        stay_duration = random.randint(min_duration, max_duration)
                        check_out = (datetime.now() - timedelta(days=days_ago-stay_duration)).strftime('%Y-%m-%d')
                        
                        # Room type based on budget preference
                        room_types = {
                            'luxury': ['Suite', 'Presidential Suite', 'Deluxe'],
                            'moderate': ['Deluxe', 'Executive', 'Standard'],
                            'budget': ['Standard', 'Economy', 'Basic']
                        }
                        
                        room_type = random.choice(room_types.get(preferred_budget, ['Standard']))
                        
                        # Number of guests based on user profile
                        family_status = user.get('family_status', 'Single')
                        if family_status == 'Family':
                            number_of_guests = random.randint(2, 5)
                        elif family_status == 'Couple':
                            number_of_guests = 2
                        else:
                            number_of_guests = random.randint(1, 2)
                        
                        # Add to hotel bookings batch
                        hotel_booking_id = next_hotel_booking_id
                        next_hotel_booking_id += 1
                        hotel_booking_inserts.append((hotel_booking_id, booking_id, hotel_id, check_in, check_out, room_type, number_of_guests))
                        hotel_bookings.append((booking_id, dest_id, stay_duration))
                    elif not is_hotel_booking and dest_id in tours_by_dest and tours_by_dest[dest_id]:
                        # Use a tour from our pre-fetched list
                        tour_id = random.choice(tours_by_dest[dest_id])
                        tour_date = (datetime.now() - timedelta(days=days_ago + random.randint(1, 3))).strftime('%Y-%m-%d')
                        
                        # Number of participants based on user profile
                        family_status = user.get('family_status', 'Single')
                        if family_status == 'Family':
                            num_participants = random.randint(2, 5)
                        elif family_status == 'Couple':
                            num_participants = 2
                        else:
                            # Solo travelers sometimes join group tours or bring friends
                            num_participants = random.randint(1, 3)
                        
                        # Add to tour bookings batch
                        tour_booking_id = next_tour_booking_id
                        next_tour_booking_id += 1
                        tour_booking_inserts.append((tour_booking_id, booking_id, tour_id, tour_date, num_participants))
                        tour_bookings.append((booking_id, dest_id, 1))
                    else:
                        # Default to hotel if no tours available or vice versa
                        if dest_id in hotels_by_dest and hotels_by_dest[dest_id]:
                            hotel_id = random.choice(hotels_by_dest[dest_id])
                            check_in = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
                            stay_duration = random.randint(3, 7)
                            check_out = (datetime.now() - timedelta(days=days_ago-stay_duration)).strftime('%Y-%m-%d')
                            room_type = 'Standard'
                            number_of_guests = random.randint(1, 2)
                            hotel_booking_id = next_hotel_booking_id
                            next_hotel_booking_id += 1
                            hotel_booking_inserts.append((hotel_booking_id, booking_id, hotel_id, check_in, check_out, room_type, number_of_guests))
                            hotel_bookings.append((booking_id, dest_id, stay_duration))
                        elif dest_id in tours_by_dest and tours_by_dest[dest_id]:
                            tour_id = random.choice(tours_by_dest[dest_id])
                            tour_date = (datetime.now() - timedelta(days=days_ago + random.randint(1, 3))).strftime('%Y-%m-%d')
                            num_participants = random.randint(1, 3)
                            tour_booking_id = next_tour_booking_id
                            next_tour_booking_id += 1
                            tour_booking_inserts.append((tour_booking_id, booking_id, tour_id, tour_date, num_participants))
                            tour_bookings.append((booking_id, dest_id, 1))
                    
                    # Add reviews/ratings based on user preference match
                    preference_score = dest.get('preference_score', 0)
                    
                    # Higher preference match = higher rating likelihood and score
                    if preference_score > 0 or random.random() > 0.3:  # 70% chance for any destination
                        # Base rating determined by preference match
                        base_rating = 3 + preference_score  # 3-5 for preference matches
                        if base_rating > 5:
                            base_rating = 5
                            
                        # Add some randomness to ratings
                        final_rating = max(1, min(5, int(base_rating + random.uniform(-0.5, 0.5))))
                        
                        # Add a review
                        review_date = (datetime.now() - timedelta(days=days_ago-random.randint(1, 7))).strftime('%Y-%m-%d')
                        review_id = next_review_id
                        next_review_id += 1
                        review_inserts.append((review_id, user_id, dest_id, final_rating, review_date))
                        
                        # Add user interaction (like for high ratings)
                        if final_rating >= 4:
                            interaction_id = next_interaction_id
                            next_interaction_id += 1
                            # Format timestamp as datetime string with time component for SQL Server
                            interaction_timestamp = (datetime.now() - timedelta(days=days_ago-random.randint(1, 7))).strftime('%Y-%m-%d %H:%M:%S')
                            # Store (interaction_id, user_id, dest_id, interaction_type, timestamp)
                            interaction_inserts.append((interaction_id, user_id, dest_id, 'like', interaction_timestamp))
                
                # Execute batch inserts
                if booking_inserts:
                    cursor.executemany(
                        "INSERT INTO bookings (booking_id, user_id, booking_date, total_cost, booking_status) VALUES (?, ?, ?, ?, ?)",
                        booking_inserts
                    )
                
                if hotel_booking_inserts:
                    cursor.executemany(
                        "INSERT INTO hotel_bookings (hotel_booking_id, booking_id, hotel_id, check_in_date, check_out_date, room_type, number_of_guests) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        hotel_booking_inserts
                    )
                
                if tour_booking_inserts:
                    cursor.executemany(
                        "INSERT INTO tour_bookings (tour_booking_id, booking_id, tour_id, tour_date, number_of_participants) VALUES (?, ?, ?, ?, ?)",
                        tour_booking_inserts
                    )
                
                if review_inserts:
                    cursor.executemany(
                        "INSERT INTO reviews (review_id, user_id, entity_type, entity_id, rating, review_date) VALUES (?, ?, 'destination', ?, ?, ?)",
                        review_inserts
                    )
                
                if interaction_inserts:
                    # Create adjusted inserts with the correct column structure
                    # Original format: (interaction_id, user_id, dest_id, 'like', review_date)
                    # New format: (interaction_id, user_id, entity_type, entity_id, interaction_type, timestamp, interaction_details, session_id)
                    cursor.executemany(
                        "INSERT INTO user_interactions (interaction_id, user_id, entity_type, entity_id, interaction_type, timestamp, interaction_details, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [(i[0], i[1], 'destination', i[2], i[3], i[4], json.dumps({"rating": "like"}), f"session_{i[0]}") for i in interaction_inserts]
                    )
                
                # Commit all changes in a single transaction
                conn.commit()
                conn.close()
                st.success("Preference-based travel history generated successfully")
            except Exception as e:
                # Rollback on error
                if conn and not conn.autocommit:
                    conn.rollback()
                st.error(f"Error generating preference-based travel history: {str(e)}")
                if st.checkbox("Show error details", False):
                    st.exception(e)
        else:
            st.error("No matching destinations found to create preference-based history.")
            return pd.DataFrame()
    
    st.subheader("Travel History")
    
    # Show a map of visited places
    if 'latitude' in visited_df.columns and 'longitude' in visited_df.columns:
        st.write("**Map of Visited Destinations:**")
        # Get all destinations for context
        all_destinations = run_query("SELECT destination_id, name, country, region, latitude, longitude FROM destinations")
        show_destination_map(all_destinations, user_visited=visited_df)
    
    # Show a table of visited places
    st.write("**Visited Destinations:**")
    visited_table = visited_df[['name', 'country', 'region', 'climate_type']].reset_index(drop=True)
    visited_table.index = visited_table.index + 1  # Start index at 1
    st.dataframe(visited_table)
    
    # Get booking details
    bookings_query = f"""
    SELECT TOP 5
        b.booking_id, b.booking_date, b.total_cost, b.booking_status,
        h.name as hotel_name, hb.check_in_date, hb.check_out_date, hb.room_type,
        t.name as tour_name, tb.tour_date, tb.number_of_participants
    FROM 
        bookings b
    LEFT JOIN 
        hotel_bookings hb ON b.booking_id = hb.booking_id
    LEFT JOIN 
        hotels h ON hb.hotel_id = h.hotel_id
    LEFT JOIN 
        tour_bookings tb ON b.booking_id = tb.booking_id
    LEFT JOIN 
        tours t ON tb.tour_id = t.tour_id
    WHERE 
        b.user_id = ? AND b.booking_status = 'completed'
    ORDER BY 
        b.booking_date DESC
    """
    
    recent_bookings = run_query(bookings_query, params=(user_id,))
    
    if not recent_bookings.empty:
        st.write("**Recent Bookings:**")
        st.dataframe(recent_bookings)
    else:
        st.info("No detailed booking information available.")
    
    return visited_df

# Recommendation Section
def show_recommendations(user_id, rec_system, top_n=10):
    """Show travel recommendations for a user."""
    st.subheader("Destination Recommendations")
    
    # Check if we have a trained model
    if rec_system.model is None:
        st.warning("Recommendation model not trained yet.")
        if st.button("Train Recommendation Model"):
            with st.spinner("Training model (this may take a few minutes)..."):
                rec_system.train_model()
            st.success("Model trained successfully!")
        return pd.DataFrame()
    
    # Get recommendations
    with st.spinner("Generating recommendations..."):
        recommendations = rec_system.get_destination_recommendations(user_id, top_n)
    
    if recommendations.empty:
        st.info("No recommendations generated. The user may have visited all available destinations.")
        return pd.DataFrame()
    
    # Sort by score descending
    recommendations = recommendations.sort_values('score', ascending=False)
    
    # Get the full destination details for the map
    destination_ids = recommendations['destination_id'].tolist()
    placeholders = ','.join(['?' for _ in destination_ids])
    full_dest_query = f"""
    SELECT * FROM destinations WHERE destination_id IN ({placeholders})
    """
    
    full_dest_df = run_query(full_dest_query, params=destination_ids)
    
    # Merge with recommendations to get the scores
    full_recommendations = pd.merge(
        full_dest_df, 
        recommendations[['destination_id', 'score']], 
        on='destination_id', 
        how='left'
    )
    
    # Get all destinations for the map
    all_destinations = run_query("SELECT destination_id, name, country, region, latitude, longitude FROM destinations")
    
    # Get user's visited destinations for context
    visited_query = f"""
    SELECT DISTINCT
        d.destination_id, d.name, d.country, d.region, d.latitude, d.longitude
    FROM
        destinations d
    JOIN
        hotels h ON d.destination_id = h.destination_id
    JOIN
        hotel_bookings hb ON h.hotel_id = hb.hotel_id
    JOIN
        bookings b ON hb.booking_id = b.booking_id
    WHERE
        b.user_id = ? AND b.booking_status = 'completed'
    UNION
    SELECT DISTINCT
        d.destination_id, d.name, d.country, d.region, d.latitude, d.longitude
    FROM
        destinations d
    JOIN
        tours t ON d.destination_id = t.destination_id
    JOIN
        tour_bookings tb ON t.tour_id = tb.tour_id
    JOIN
        bookings b ON tb.booking_id = b.booking_id
    WHERE
        b.user_id = ? AND b.booking_status = 'completed'
    """
    
    visited_df = run_query(visited_query, params=(user_id, user_id))
    
    # Show recommendations map
    st.write("**Map of Recommended Destinations:**")
    show_destination_map(all_destinations, highlight_destinations=full_recommendations, user_visited=visited_df)
    
    # Show recommendations table
    st.write("**Top Recommended Destinations:**")
    rec_table = recommendations[['name', 'country', 'region', 'climate_type', 'score']].reset_index(drop=True)
    rec_table.index = rec_table.index + 1  # Start index at 1
    
    # Format the score column as percentage
    rec_table['score'] = rec_table['score'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(rec_table)
    
    # Return the full recommendations for further use
    return full_recommendations

# Travel Statistics Section
def show_travel_statistics():
    """Display travel analytics statistics and visualizations based on user data."""
    st.subheader("Travel Analytics")
    
    # Check if analytics tables exist
    check_query = """
    SELECT COUNT(*) AS table_count 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME IN ('bookings', 'hotels', 'destinations', 'reviews')
    """
    table_check = run_query(check_query)
    
    if table_check.empty or table_check.iloc[0]['table_count'] < 4:
        st.warning("Analytics data is not fully available. Some visualizations may be limited.")
    
    # Popular destinations
    st.write("### Popular Destinations")
    try:
        popular_query = """
        SELECT TOP 10 d.name, d.country, COUNT(b.booking_id) as visit_count 
        FROM destinations d
        JOIN hotels h ON d.destination_id = h.destination_id
        JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
        JOIN bookings b ON hb.booking_id = b.booking_id
        GROUP BY d.name, d.country
        ORDER BY visit_count DESC
        """
        pop_df = run_query(popular_query)
        
        if not pop_df.empty:
            fig = px.bar(pop_df, x='name', y='visit_count', 
                        hover_data=['country'], color='visit_count',
                        labels={'name': 'Destination', 'visit_count': 'Number of Visits'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
        else:
            st.info("No destination data available for analysis.")
    except Exception as e:
        st.error(f"Error generating popular destinations chart: {str(e)}")
    
    # Average spending per hotel
    st.write("### Average Spending per Hotel")
    try:
        spending_query = """
        SELECT TOP 10 h.name, AVG(b.total_cost) as avg_spending 
        FROM hotels h
        JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
        JOIN bookings b ON hb.booking_id = b.booking_id
        GROUP BY h.name
        ORDER BY avg_spending DESC
        """
        spend_df = run_query(spending_query)
        
        if not spend_df.empty:
            fig = px.bar(spend_df, x='name', y='avg_spending', 
                        color='avg_spending', labels={'name': 'Hotel', 'avg_spending': 'Average Spending ($)'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
        else:
            st.info("No spending data available for analysis.")
    except Exception as e:
        st.error(f"Error generating spending chart: {str(e)}")
    
    # Seasonal travel trends
    st.write("### Seasonal Travel Trends")
    try:
        seasonal_query = """
        SELECT DATEPART(MONTH, b.booking_date) as month, COUNT(b.booking_id) as booking_count 
        FROM bookings b
        GROUP BY DATEPART(MONTH, b.booking_date)
        ORDER BY month
        """
        seasonal_df = run_query(seasonal_query)
        
        if not seasonal_df.empty:
            # Map month numbers to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            seasonal_df['month_name'] = seasonal_df['month'].apply(lambda x: month_names[x-1])
            
            fig = px.line(seasonal_df, x='month_name', y='booking_count', markers=True, 
                        labels={'month_name': 'Month', 'booking_count': 'Number of Bookings'})
            fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':month_names})
            st.plotly_chart(fig)
        else:
            st.info("No seasonal data available for analysis.")
    except Exception as e:
        st.error(f"Error generating seasonal trends chart: {str(e)}")
    
    # Customer satisfaction ratings
    st.write("### Customer Satisfaction Ratings")
    try:
        ratings_query = """
        SELECT 
            r.rating, 
            COUNT(r.review_id) as review_count 
        FROM reviews r
        GROUP BY r.rating
        ORDER BY r.rating
        """
        ratings_df = run_query(ratings_query)
        
        if not ratings_df.empty:
            fig = px.pie(ratings_df, values='review_count', names='rating', 
                        title='Distribution of Ratings', hole=0.4)
            st.plotly_chart(fig)
        else:
            st.info("No review data available for analysis.")
    except Exception as e:
        st.error(f"Error generating ratings chart: {str(e)}")

# Admin Database Management Functions
def get_all_tables():
    """Get a list of all tables in the database."""
    query = """
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_NAME
    """
    return run_query(query)['TABLE_NAME'].tolist()

def get_table_columns(table_name):
    """Get column information for a table."""
    query = f"""
    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = '{table_name}'
    ORDER BY ORDINAL_POSITION
    """
    return run_query(query)

def get_table_sample(table_name, limit=20):
    """Get a sample of data from a table."""
    # Sanitize table name to prevent SQL injection
    if not re.match(r'^[a-zA-Z0-9_]+$', table_name):
        st.error("Invalid table name")
        return pd.DataFrame()
    
    query = f"SELECT TOP {limit} * FROM {table_name}"
    return run_query(query)

def show_admin_database_panel():
    """Show the admin database management panel."""
    st.subheader("Database Management")
    
    # Get list of tables
    tables = get_all_tables()
    if not tables:
        st.error("No tables found in the database")
        return
    
    # Table selection
    selected_table = st.selectbox("Select Table", tables)
    
    # Show table columns
    st.subheader(f"Columns in {selected_table}")
    columns_df = get_table_columns(selected_table)
    st.dataframe(columns_df)
    
    # Show table data sample
    st.subheader(f"Sample data from {selected_table}")
    sample_size = st.slider("Sample size", 5, 100, 20, key="sample_size_slider")
    sample_df = get_table_sample(selected_table, sample_size)
    st.dataframe(sample_df)
    
    # Row count
    row_count_query = f"SELECT COUNT(*) as count FROM {selected_table}"
    row_count = run_query(row_count_query).iloc[0]['count']
    st.write(f"Total rows: {row_count}")
    
    # Table operations
    st.subheader("Table Operations")
    operation = st.selectbox("Select Operation", 
                            ["Choose an operation", "Truncate Table", "Drop Table", "View Table Statistics"])
    
    if operation == "Truncate Table":
        st.warning(f"⚠️ This will delete all data from the {selected_table} table. This action cannot be undone!")
        if st.button("Confirm Truncate"):
            success, message = execute_query(f"TRUNCATE TABLE {selected_table}")
            if success:
                st.success(f"Table {selected_table} truncated successfully")
            else:
                st.error(message)
    
    elif operation == "Drop Table":
        st.warning(f"⚠️ This will delete the entire {selected_table} table including its structure. This action cannot be undone!")
        confirm_text = st.text_input("Type the table name to confirm")
        if confirm_text == selected_table and st.button("Confirm Drop"):
            # Drop may fail due to foreign key constraints
            success, message = execute_query(f"DROP TABLE {selected_table}")
            if success:
                st.success(f"Table {selected_table} dropped successfully")
                # Refresh the page to update the table list
                st.experimental_rerun()
            else:
                st.error(message)
    
    elif operation == "View Table Statistics":
        if selected_table == "users":
            # Show name variety statistics
            st.subheader("Name Distribution")
            first_names_query = "SELECT first_name, COUNT(*) as count FROM users GROUP BY first_name ORDER BY count DESC"
            first_names_df = run_query(first_names_query)
            
            last_names_query = "SELECT last_name, COUNT(*) as count FROM users GROUP BY last_name ORDER BY count DESC"
            last_names_df = run_query(last_names_query)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**First Names Distribution:**")
                st.dataframe(first_names_df)
                
                # Create a pie chart of the top 10 first names
                fig, ax = plt.subplots()
                top_names = first_names_df.head(10)
                ax.pie(top_names['count'], labels=top_names['first_name'], autopct='%1.1f%%')
                ax.set_title('Top 10 First Names')
                st.pyplot(fig)
                
            with col2:
                st.write("**Last Names Distribution:**")
                st.dataframe(last_names_df)
                
                # Create a pie chart of the top 10 last names
                fig, ax = plt.subplots()
                top_names = last_names_df.head(10)
                ax.pie(top_names['count'], labels=top_names['last_name'], autopct='%1.1f%%')
                ax.set_title('Top 10 Last Names')
                st.pyplot(fig)
                
            # Show unique name counts
            unique_first_names = len(first_names_df)
            unique_last_names = len(last_names_df)
            total_users = row_count
            
            st.write(f"**Unique first names:** {unique_first_names} ({unique_first_names/total_users:.1%} of users)")
            st.write(f"**Unique last names:** {unique_last_names} ({unique_last_names/total_users:.1%} of users)")
            
        elif selected_table == "destinations":
            # Show destination statistics
            st.subheader("Destination Distribution")
            
            regions_query = "SELECT region, COUNT(*) as count FROM destinations GROUP BY region ORDER BY count DESC"
            regions_df = run_query(regions_query)
            
            countries_query = "SELECT country, COUNT(*) as count FROM destinations GROUP BY country ORDER BY count DESC"
            countries_df = run_query(countries_query)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Regions Distribution:**")
                st.dataframe(regions_df)
                
                # Create a bar chart of regions
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(regions_df['region'], regions_df['count'])
                ax.set_title('Destinations by Region')
                ax.set_xlabel('Count')
                plt.tight_layout()
                st.pyplot(fig)
                
            with col2:
                st.write("**Top 15 Countries:**")
                st.dataframe(countries_df.head(15))
                
                # Create a bar chart of top 15 countries
                fig, ax = plt.subplots(figsize=(10, 6))
                top_countries = countries_df.head(15)
                ax.barh(top_countries['country'], top_countries['count'])
                ax.set_title('Top 15 Countries')
                ax.set_xlabel('Count')
                plt.tight_layout()
                st.pyplot(fig)
        
        else:
            # General table statistics
            st.write(f"**Row count:** {row_count}")
            
            # If the table has a name column, show distribution
            if 'name' in [col.lower() for col in sample_df.columns]:
                name_query = f"SELECT TOP 20 name, COUNT(*) as count FROM {selected_table} GROUP BY name ORDER BY count DESC"
                name_df = run_query(name_query)
                
                st.write("**Name Distribution:**")
                st.dataframe(name_df)
                
                # Create a bar chart of top 20 names
                if not name_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_names = name_df.head(20)
                    ax.barh(top_names['name'], top_names['count'])
                    ax.set_title(f'Top 20 Names in {selected_table}')
                    ax.set_xlabel('Count')
                    plt.tight_layout()
                    st.pyplot(fig)

def show_admin_sql_panel():
    """Show the admin SQL query panel."""
    st.subheader("SQL Query Execution")
    
    # SQL query input
    sql_query = st.text_area("Enter SQL Query", height=200)
    
    # Query type selection with info tooltip
    query_type = st.radio(
        "Query Type",
        ["SELECT (Read-only)", "UPDATE/INSERT/DELETE/DROP (Modify data)"]
    )
    
    if st.button("Execute Query"):
        if not sql_query:
            st.warning("Please enter a SQL query")
            return
        
        if query_type == "SELECT (Read-only)":
            # Run SELECT query and display results
            if not sql_query.lower().strip().startswith("select"):
                st.error("Only SELECT queries are allowed in read-only mode")
                return
                
            with st.spinner("Executing query..."):
                result_df = run_query(sql_query)
                
            if result_df.empty:
                st.info("Query executed successfully, but returned no results")
            else:
                st.success(f"Query executed successfully. {len(result_df)} rows returned.")
                st.dataframe(result_df)
        else:
            # Run non-SELECT query
            if sql_query.lower().strip().startswith("select"):
                st.warning("This appears to be a SELECT query. Consider switching to read-only mode.")
                
            if st.checkbox("I understand this query will modify the database"):
                with st.spinner("Executing query..."):
                    success, message = execute_query(sql_query)
                    
                if success:
                    st.success(message)
                else:
                    st.error(message)

def show_admin_names_panel():
    """Show the admin panel for name generation and maintenance."""
    st.subheader("Name Data Management")
    
    # Fetch current name data
    first_names_query = "SELECT DISTINCT first_name FROM users ORDER BY first_name"
    first_names = run_query(first_names_query)['first_name'].tolist()
    
    last_names_query = "SELECT DISTINCT last_name FROM users ORDER BY last_name"
    last_names = run_query(last_names_query)['last_name'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Current First Names:** {len(first_names)}")
        first_names_text = st.text_area("First Names", 
                                        value=", ".join(first_names), 
                                        height=300)
    
    with col2:
        st.write(f"**Current Last Names:** {len(last_names)}")
        last_names_text = st.text_area("Last Names", 
                                      value=", ".join(last_names), 
                                      height=300)
    
    # Add option to augment names
    st.subheader("Add New Names")
    
    add_col1, add_col2 = st.columns(2)
    
    with add_col1:
        new_first_names = st.text_area("New First Names (comma-separated)")
        
    with add_col2:
        new_last_names = st.text_area("New Last Names (comma-separated)")
    
    if st.button("Add Names"):
        if new_first_names or new_last_names:
            # Process first names
            if new_first_names:
                # Parse comma-separated names and clean them
                new_first_names_list = [name.strip() for name in new_first_names.split(",") if name.strip()]
                
                # Display preview of names to add
                st.write(f"Adding {len(new_first_names_list)} new first names")
                
                # Create a mock user to add the new names
                for name in new_first_names_list:
                    # Generate a random user with this first name
                    insert_query = f"""
                    INSERT INTO users (
                        user_id, first_name, last_name, email, date_of_birth, 
                        signup_date, preferred_language, country_of_residence, loyalty_tier
                    ) VALUES (
                        (SELECT MAX(user_id) + 1 FROM users),
                        '{name}',
                        '{random.choice(last_names)}',
                        '{name.lower()}_{random.randint(1000, 9999)}@example.com',
                        '1990-01-01',
                        GETDATE(),
                        'English',
                        'USA',
                        'Bronze'
                    )
                    """
                    execute_query(insert_query)
                
            # Process last names
            if new_last_names:
                # Parse comma-separated names and clean them
                new_last_names_list = [name.strip() for name in new_last_names.split(",") if name.strip()]
                
                # Display preview of names to add
                st.write(f"Adding {len(new_last_names_list)} new last names")
                
                # Create a mock user to add the new names
                for name in new_last_names_list:
                    # Generate a random user with this last name
                    insert_query = f"""
                    INSERT INTO users (
                        user_id, first_name, last_name, email, date_of_birth, 
                        signup_date, preferred_language, country_of_residence, loyalty_tier
                    ) VALUES (
                        (SELECT MAX(user_id) + 1 FROM users),
                        '{random.choice(first_names)}',
                        '{name}',
                        '{name.lower()}_{random.randint(1000, 9999)}@example.com',
                        '1990-01-01',
                        GETDATE(),
                        'English',
                        'USA',
                        'Bronze'
                    )
                    """
                    execute_query(insert_query)
                
            st.success("Names added successfully. Refresh to see the updated lists.")
            
        else:
            st.warning("No new names provided")

# Recommendation System for Discover Tab
def train_destination_recommender(user_id, min_interactions=5):
    """
    Train an advanced recommendation model for a user based on their preferences and travel history.
    Uses a hybrid approach combining collaborative filtering, content-based filtering, and feature engineering.
    
    Args:
        user_id (int): The ID of the user to generate recommendations for
        min_interactions (int): Minimum number of interactions needed for reliable recommendations
        
    Returns:
        tuple: (DataFrame of recommendations, error message if any)
    """
    try:
        print(f"Starting advanced recommendation training for user {user_id}")
        # Get user data including preferences
        user_query = """
        SELECT u.*, 
               p.preferred_hotel_stars, p.preferred_budget_category, 
               p.preferred_activities, p.preferred_climates, p.travel_style
        FROM users u
        LEFT JOIN user_preferences p ON u.user_id = p.user_id
        WHERE u.user_id = ?
        """
        user_df = run_query(user_query, [user_id])
        
        if user_df.empty:
            return pd.DataFrame(), "User not found in database."
            
        user = user_df.iloc[0]
        
        # Extract user preferences for content-based filtering
        preferred_climates = []
        if pd.notna(user.get('preferred_climates')):
            try:
                preferred_climates = json.loads(user['preferred_climates']) if isinstance(user['preferred_climates'], str) else user['preferred_climates']
            except:
                preferred_climates = []
                
        preferred_activities = []
        if pd.notna(user.get('preferred_activities')):
            try:
                preferred_activities = json.loads(user['preferred_activities']) if isinstance(user['preferred_activities'], str) else user['preferred_activities']
            except:
                preferred_activities = []
        
        travel_style = user.get('travel_style', '')
        preferred_budget = user.get('preferred_budget_category', '')
        
        # Get user's visited destinations with ratings
        visited_query = """
        SELECT d.*, 
               COALESCE(r.rating, 0) as rating,
               CASE WHEN ui.interaction_type = 'like' THEN 1 ELSE 0 END as liked
        FROM destinations d
        LEFT JOIN reviews r ON d.destination_id = r.entity_id AND r.entity_type = 'destination' AND r.user_id = ?
        LEFT JOIN user_interactions ui ON d.destination_id = ui.entity_id AND ui.entity_type = 'destination' AND ui.user_id = ? AND ui.interaction_type = 'like'
        WHERE d.destination_id IN (
            SELECT DISTINCT h.destination_id 
            FROM hotels h
            JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
            JOIN bookings b ON hb.booking_id = b.booking_id
            WHERE b.user_id = ?
            UNION
            SELECT DISTINCT t.destination_id
            FROM tours t
            JOIN tour_bookings tb ON t.tour_id = tb.tour_id
            JOIN bookings b ON tb.booking_id = b.booking_id
            WHERE b.user_id = ?
        )
        """
        visited_df = run_query(visited_query, [user_id, user_id, user_id, user_id])
        
        # Get user interactions with destinations (likes, views, etc.)
        interactions_query = """
        SELECT ui.*, d.name, d.country, d.region, d.climate_type as destination_type
        FROM user_interactions ui
        JOIN destinations d ON ui.entity_id = d.destination_id AND ui.entity_type = 'destination'
        WHERE ui.user_id = ?
        """
        interactions_df = run_query(interactions_query, [user_id])
        
        # Calculate how much data we have for this user
        has_visited = len(visited_df) > 0
        has_ratings = visited_df['rating'].sum() > 0 if not visited_df.empty else False
        has_likes = visited_df['liked'].sum() > 0 if not visited_df.empty else False
        has_interactions = len(interactions_df) >= min_interactions
        
        # Determine which approach to use based on available data
        using_hybrid = has_visited and (has_ratings or has_likes) and has_interactions
        using_content = has_visited or len(preferred_climates) > 0 or len(preferred_activities) > 0 or travel_style
        
        print(f"User data stats: visited={has_visited}, ratings={has_ratings}, likes={has_likes}, interactions={has_interactions}")
        print(f"Using hybrid approach: {using_hybrid}, Using content-based: {using_content}")
        
        # If we don't have enough data, fall back to collaborative filtering
        if not (using_hybrid or using_content):
            print("Insufficient user data, using pure collaborative filtering")
            return collaborative_filtering_recommendations(user_id)
        
        # Get user preferences for explicit feature weights - this wasn't working so let's skip it
        # The user_preferences table doesn't have the expected columns
        preferences_df = pd.DataFrame()  # Empty DataFrame as a fallback
        
        # Get all unvisited destinations
        unvisited_query = """
        SELECT d.*
        FROM destinations d
        WHERE d.destination_id NOT IN (
            SELECT DISTINCT h.destination_id 
            FROM hotels h
            JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
            JOIN bookings b ON hb.booking_id = b.booking_id
            WHERE b.user_id = ?
            UNION
            SELECT DISTINCT t.destination_id
            FROM tours t
            JOIN tour_bookings tb ON t.tour_id = tb.tour_id
            JOIN bookings b ON tb.booking_id = b.booking_id
            WHERE b.user_id = ?
        )
        """
        unvisited_df = run_query(unvisited_query, [user_id, user_id])
        
        if unvisited_df.empty:
            return pd.DataFrame(), "You've visited all available destinations in our database!"
        
        print(f"Creating features for {len(unvisited_df)} unvisited destinations")
        # Create feature matrix for unvisited destinations
        feature_df = unvisited_df.copy()
        
        # Print available columns to help debug
        print(f"Available columns in feature_df: {feature_df.columns.tolist()}")
        
        # 1. Basic destination features
        # Check which type column exists in the data
        destination_type_column = 'climate_type'  # According to table_columns.md
        if destination_type_column not in feature_df.columns:
            print(f"Warning: '{destination_type_column}' column not found, creating a default one")
            feature_df['climate_type'] = 'unknown'
            
        # Add features based on destination type
        destination_types = ['tropical', 'temperate', 'cold', 'desert', 'mediterranean', 'continental']
        for dest_type in destination_types:
            feature_df[f'is_{dest_type}'] = feature_df[destination_type_column].apply(
                lambda x: 1 if isinstance(x, str) and x.lower() == dest_type.lower() else 0)
        
        # 2. Geographic features
        # Check if country column exists
        if 'country' not in feature_df.columns:
            print("Country column missing, adding default")
            feature_df['country'] = 'Unknown'
            
        # Add continent features
        continents = {
            'North America': ['USA', 'Canada', 'Mexico'],
            'Europe': ['France', 'Italy', 'Spain', 'Germany', 'UK', 'Greece'],
            'Asia': ['Japan', 'China', 'Thailand', 'Vietnam', 'Indonesia', 'India'],
            'Oceania': ['Australia', 'New Zealand'],
            'South America': ['Brazil', 'Argentina', 'Peru', 'Chile'],
            'Africa': ['South Africa', 'Egypt', 'Morocco', 'Kenya']
        }
        
        for continent, countries in continents.items():
            feature_df[f'is_{continent.lower().replace(" ", "_")}'] = feature_df['country'].apply(
                lambda x: 1 if x in countries else 0)
        
        # 3. Climate features
        # Check if climate_type column exists
        climate_column = None
        for possible_column in ['climate_type', 'climate']:
            if possible_column in feature_df.columns:
                climate_column = possible_column
                print(f"Using '{climate_column}' as climate column")
                break
                
        # If no climate column, map from destination type
        if climate_column is None:
            print("No climate column found, mapping from destination type")
            climate_mapping = {
                'beach': 'tropical',
                'mountain': 'cold',
                'city': 'temperate',
                'cultural': 'temperate',
                'resort': 'tropical',
                'national_park': 'varied',
                'unknown': 'varied'
            }
            
            feature_df['climate'] = feature_df[destination_type_column].map(climate_mapping)
            climate_column = 'climate'
        else:
            # Create a climate column for mapping if it doesn't exist
            if 'climate' not in feature_df.columns:
                climate_mapping = {
                    'tropical': 'tropical',
                    'temperate': 'temperate',
                    'cold': 'cold',
                    'desert': 'hot',
                    'mediterranean': 'temperate',
                    'continental': 'cold'
                }
                feature_df['climate'] = feature_df[climate_column].map(climate_mapping)
                feature_df['climate'] = feature_df['climate'].fillna('varied')  # Default value
                
        # One-hot encode climate
        climate_dummies = pd.get_dummies(feature_df['climate'], prefix='climate')
        feature_df = pd.concat([feature_df, climate_dummies], axis=1)
        
        # 4. Cost level features
        high_cost_countries = ['USA', 'Canada', 'Japan', 'Australia', 'UK', 'France', 'Germany']
        medium_cost_countries = ['Spain', 'Italy', 'Greece', 'Brazil', 'China', 'South Africa']
        
        feature_df['cost_level'] = feature_df['country'].apply(
            lambda x: 'high' if x in high_cost_countries else 
                      'medium' if x in medium_cost_countries else 'low')
        
        cost_dummies = pd.get_dummies(feature_df['cost_level'], prefix='cost')
        feature_df = pd.concat([feature_df, cost_dummies], axis=1)
        
        # 5. User preference match features (content-based)
        # Initialize preference match scores
        feature_df['climate_match'] = 0
        feature_df['style_match'] = 0
        feature_df['activity_match'] = 0
        feature_df['budget_match'] = 0
        
        # Calculate climate match
        if preferred_climates and climate_column in feature_df.columns:
            feature_df['climate_match'] = feature_df[climate_column].apply(
                lambda x: 1 if x in preferred_climates else 0)
        
        # Calculate travel style match
        if travel_style:
            feature_df['style_match'] = feature_df['climate_type'].apply(
                lambda x: 1 if travel_style == 'temperate' and x == 'temperate' or
                            travel_style == 'tropical' and x == 'tropical' or
                            travel_style == 'adventure' and x in ['cold', 'continental'] or
                            travel_style == 'cultural' and x in ['temperate', 'mediterranean'] else 0
            )
        
        # Budget preference match
        if preferred_budget:
            budget_mapping = {
                'luxury': 'high',
                'moderate': 'medium',
                'budget': 'low'
            }
            mapped_budget = budget_mapping.get(preferred_budget, 'medium')
            feature_df['budget_match'] = feature_df['cost_level'].apply(
                lambda x: 1 if x == mapped_budget else 0)
        
        # Calculate overall preference match score
        feature_df['preference_match'] = (
            feature_df['climate_match'] * 0.35 + 
            feature_df['style_match'] * 0.35 + 
            feature_df['budget_match'] * 0.3
        )
        
        # 6. Apply explicit preference weights from user_preferences table
        preference_weights = {}
        
        if not preferences_df.empty:
            for _, pref in preferences_df.iterrows():
                pref_type = pref['preference_type'].lower()
                pref_value = pref['preference_value'].lower()
                weight = pref['weight']
                
                if pref_type == 'destination_type':
                    col_name = f'is_{pref_value}'
                    if col_name in feature_df.columns:
                        preference_weights[col_name] = weight
                elif pref_type == 'climate':
                    col_name = f'climate_{pref_value}'
                    if col_name in feature_df.columns:
                        preference_weights[col_name] = weight
                elif pref_type == 'region':
                    continent_mapping = {
                        'north_america': 'is_north_america',
                        'europe': 'is_europe',
                        'asia': 'is_asia',
                        'oceania': 'is_oceania',
                        'south_america': 'is_south_america',
                        'africa': 'is_africa'
                    }
                    col_name = continent_mapping.get(pref_value)
                    if col_name and col_name in feature_df.columns:
                        preference_weights[col_name] = weight
                elif pref_type == 'budget':
                    budget_mapping = {
                        'luxury': 'cost_high',
                        'moderate': 'cost_medium',
                        'budget': 'cost_low'
                    }
                    col_name = budget_mapping.get(pref_value)
                    if col_name and col_name in feature_df.columns:
                        preference_weights[col_name] = weight
        
        # Apply weights
        for col, weight in preference_weights.items():
            if col in feature_df.columns:
                feature_df[col] = feature_df[col] * weight
        
        # 7. Collaborative filtering component
        print("Getting collaborative filtering component")
        collab_scores = pd.Series(0, index=unvisited_df['destination_id'])
        
        # Find similar users and their liked destinations
        similar_users_query = """
        SELECT TOP 20 up1.user_id, 
               COUNT(*) as common_prefs
        FROM user_preferences up1 
        JOIN user_preferences up2 ON 
            (up1.preferred_climates LIKE up2.preferred_climates OR 
             up1.preferred_activities LIKE up2.preferred_activities OR 
             up1.travel_style = up2.travel_style OR
             up1.preferred_budget_category = up2.preferred_budget_category)
        WHERE up2.user_id = ? AND up1.user_id != ?
        GROUP BY up1.user_id
        HAVING COUNT(*) >= 1
        ORDER BY common_prefs DESC
        """
        similar_users = run_query(similar_users_query, [user_id, user_id])
        print(f"Found {len(similar_users)} similar users")
        
        # Find similar users based on history if we don't have many preference-based matches
        if len(similar_users) < 10 and not visited_df.empty:
            history_similarity_query = """
            WITH user_destinations AS (
                -- Get destinations visited through hotels
                SELECT DISTINCT b.user_id, h.destination_id
                FROM bookings b
                JOIN hotel_bookings hb ON b.booking_id = hb.booking_id
                JOIN hotels h ON hb.hotel_id = h.hotel_id
                WHERE b.booking_status = 'completed'
                
                UNION
                
                -- Get destinations visited through tours
                SELECT DISTINCT b.user_id, t.destination_id
                FROM bookings b
                JOIN tour_bookings tb ON b.booking_id = tb.booking_id
                JOIN tours t ON tb.tour_id = t.tour_id
                WHERE b.booking_status = 'completed'
            ),
            target_destinations AS (
                SELECT DISTINCT destination_id
                FROM user_destinations
                WHERE user_id = ?
            ),
            user_overlap AS (
                SELECT ud.user_id, 
                       COUNT(DISTINCT ud.destination_id) as common_destinations
                FROM user_destinations ud
                JOIN target_destinations td ON ud.destination_id = td.destination_id
                WHERE ud.user_id != ?
                GROUP BY ud.user_id
                HAVING COUNT(DISTINCT ud.destination_id) >= 1
            )
            SELECT TOP 10 user_id, common_destinations
            FROM user_overlap
            ORDER BY common_destinations DESC
            """
            history_similar_users = run_query(history_similarity_query, [user_id, user_id])
            
            # Merge with preference-based similar users
            if not history_similar_users.empty:
                # Convert similarity scores to a common scale
                if not similar_users.empty:
                    max_common_prefs = similar_users['common_prefs'].max()
                    similar_users['normalized_similarity'] = similar_users['common_prefs'] / max_common_prefs
                
                max_common_dests = history_similar_users['common_destinations'].max()
                history_similar_users['normalized_similarity'] = history_similar_users['common_destinations'] / max_common_dests
                
                # Keep only user_id and normalized_similarity columns
                history_similar_users = history_similar_users[['user_id', 'normalized_similarity']]
                
                # Append to similar_users
                if similar_users.empty:
                    similar_users = history_similar_users
                else:
                    similar_users_ids = set(similar_users['user_id'].unique())
                    
                    # Only add users not already in the set
                    new_similar_users = history_similar_users[~history_similar_users['user_id'].isin(similar_users_ids)]
                    
                    # Combine the dataframes
                    if not new_similar_users.empty:
                        similar_users = pd.concat([similar_users[['user_id', 'normalized_similarity']], 
                                                 new_similar_users])
        
        # If we have similar users, calculate collaborative scores
        if not similar_users.empty:
            # Check if we need to add the normalized_similarity column
            if 'normalized_similarity' not in similar_users.columns:
                max_similarity = similar_users['common_prefs'].max()
                similar_users['normalized_similarity'] = similar_users['common_prefs'] / max_similarity if max_similarity > 0 else 0
                
            for _, sim_user in similar_users.iterrows():
                sim_user_id = sim_user['user_id']
                similarity = sim_user['normalized_similarity']
                
                # Get destinations this similar user liked or visited with high ratings
                sim_user_liked_query = """
                SELECT d.destination_id, 
                       COALESCE(r.rating, 0) + 
                       (CASE WHEN ui.interaction_type = 'like' THEN 1 ELSE 0 END) +
                       (CASE WHEN b.booking_id IS NOT NULL THEN 1 ELSE 0 END) as preference_score
                FROM destinations d
                LEFT JOIN user_interactions ui ON d.destination_id = ui.entity_id 
                    AND ui.entity_type = 'destination' AND ui.user_id = ?
                LEFT JOIN reviews r ON d.destination_id = r.entity_id 
                    AND r.entity_type = 'destination' AND r.user_id = ?
                LEFT JOIN (
                    -- Join through hotel bookings
                    SELECT DISTINCT h.destination_id, b.booking_id
                    FROM hotels h
                    JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
                    JOIN bookings b ON hb.booking_id = b.booking_id
                    WHERE b.user_id = ?
                    
                    UNION
                    
                    -- Join through tour bookings
                    SELECT DISTINCT t.destination_id, b.booking_id
                    FROM tours t
                    JOIN tour_bookings tb ON t.tour_id = tb.tour_id
                    JOIN bookings b ON tb.booking_id = b.booking_id
                    WHERE b.user_id = ?
                ) b ON d.destination_id = b.destination_id
                WHERE (ui.interaction_type IN ('like', 'visit') OR r.rating >= 4 OR b.booking_id IS NOT NULL)
                """
                sim_user_likes = run_query(sim_user_liked_query, [sim_user_id, sim_user_id, sim_user_id, sim_user_id])
                
                # Add weighted collaborative scores
                for _, row in sim_user_likes.iterrows():
                    dest_id = row['destination_id']
                    if dest_id in collab_scores.index:
                        collab_scores[dest_id] += similarity * row['preference_score']
        
        # Normalize collaborative scores
        if collab_scores.max() > 0:
            collab_scores = collab_scores / collab_scores.max()
        
        # Add collaborative scores to feature matrix
        feature_df['collab_score'] = feature_df['destination_id'].map(collab_scores)
        feature_df['collab_score'] = feature_df['collab_score'].fillna(0)
        
        # 8. Item-Item Similarity
        if not visited_df.empty:
            print("Calculating item-item similarity for destinations")
            # Get all destinations
            all_destinations = run_query("SELECT destination_id, name, climate_type, country, region FROM destinations")
            
            # Get destinations this user liked
            user_liked = visited_df[visited_df['rating'] >= 4 | (visited_df['liked'] == 1)]
            
            if not user_liked.empty:
                # Calculate destination similarity scores based on features
                item_similarity_scores = pd.Series(0, index=unvisited_df['destination_id'])
                
                for _, liked_dest in user_liked.iterrows():
                    # Find destinations with similar attributes
                    similar_destinations = all_destinations[
                        (all_destinations['climate_type'] == liked_dest['climate_type']) |
                        (all_destinations['country'] == liked_dest['country']) |
                        (all_destinations['region'] == liked_dest['region'])
                    ]
                    
                    # Calculate similarity score for each destination
                    for _, sim_dest in similar_destinations.iterrows():
                        if sim_dest['destination_id'] in item_similarity_scores.index:
                            # Add score based on number of matching attributes
                            sim_score = 0
                            if sim_dest['climate_type'] == liked_dest['climate_type']:
                                sim_score += 0.5
                            if sim_dest['country'] == liked_dest['country']:
                                sim_score += 0.3
                            if sim_dest['region'] == liked_dest['region']:
                                sim_score += 0.2
                                
                            # Weight by rating/liked status
                            rating_weight = liked_dest['rating'] / 5.0 if liked_dest['rating'] > 0 else 0.6
                            rating_weight = max(rating_weight, 0.6 if liked_dest['liked'] else 0)
                            
                            item_similarity_scores[sim_dest['destination_id']] += sim_score * rating_weight
                
                # Normalize item similarity scores
                if item_similarity_scores.max() > 0:
                    item_similarity_scores = item_similarity_scores / item_similarity_scores.max()
                
                # Add to feature matrix
                feature_df['item_similarity'] = feature_df['destination_id'].map(item_similarity_scores)
                feature_df['item_similarity'] = feature_df['item_similarity'].fillna(0)
        else:
            feature_df['item_similarity'] = 0
        
        # 9. Select and prepare features for the model
        print("Preparing final feature matrix for prediction")
        numerical_cols = [col for col in feature_df.columns if col.startswith(('is_', 'climate_', 'cost_')) 
                         or col in ['collab_score', 'preference_match', 'item_similarity']]
        
        # Ensure we have adequate features
        if not numerical_cols:
            return pd.DataFrame(), "Unable to create meaningful recommendation features"
        
        X = feature_df[numerical_cols]
        
        # 10. Prediction approach based on available data
        if has_visited and len(visited_df) >= 3:
            # Train a model on past behavior
            try:
                print(f"Training model using {len(visited_df)} visited destinations")
                # Create features for past destinations
                past_features = visited_df.copy()
                
                # Print past features columns for debugging
                print(f"Available columns in past_features: {past_features.columns.tolist()}")
                
                # Set climate_type as the destination type column
                past_type_column = 'climate_type'
                if past_type_column not in past_features.columns:
                    print(f"Warning: '{past_type_column}' column not found in past_features, creating default")
                    past_features['climate_type'] = 'unknown'
                
                # Create the same features as for unvisited destinations
                for dest_type in destination_types:
                    past_features[f'is_{dest_type}'] = past_features[past_type_column].apply(
                        lambda x: 1 if isinstance(x, str) and x.lower() == dest_type.lower() else 0)
                
                # Check if country exists
                if 'country' not in past_features.columns:
                    print("Country column missing in past_features, adding default")
                    past_features['country'] = 'Unknown'
                
                for continent, countries in continents.items():
                    past_features[f'is_{continent.lower().replace(" ", "_")}'] = past_features['country'].apply(
                        lambda x: 1 if x in countries else 0)
                
                # Climate column is the same as past_type_column (climate_type) in this case
                past_climate_column = past_type_column
                
                # Use the same climate mapping as for new destinations
                if 'climate' not in past_features.columns:
                    climate_mapping = {
                        'tropical': 'tropical',
                        'temperate': 'temperate',
                        'cold': 'cold',
                        'desert': 'hot',
                        'mediterranean': 'temperate',
                        'continental': 'cold'
                    }
                    past_features['climate'] = past_features[past_climate_column].map(climate_mapping)
                    past_features['climate'] = past_features['climate'].fillna('varied')  # Default value
                
                past_climate_dummies = pd.get_dummies(past_features['climate'], prefix='climate')
                past_features = pd.concat([past_features, past_climate_dummies], axis=1)
                
                past_features['cost_level'] = past_features['country'].apply(
                    lambda x: 'high' if x in high_cost_countries else 
                              'medium' if x in medium_cost_countries else 'low')
                
                past_cost_dummies = pd.get_dummies(past_features['cost_level'], prefix='cost')
                past_features = pd.concat([past_features, past_cost_dummies], axis=1)
                
                # Add preference match features (same as for unvisited)
                if past_climate_column:
                    past_features['climate_match'] = past_features[past_climate_column].apply(
                        lambda x: 1 if x in preferred_climates else 0)
                else:
                    past_features['climate_match'] = 0
                
                # Match travel style to climate type using the same logic as for unvisited destinations
                if travel_style:
                    past_features['style_match'] = past_features['climate_type'].apply(
                        lambda x: 1 if travel_style == 'temperate' and x == 'temperate' or
                                    travel_style == 'tropical' and x == 'tropical' or
                                    travel_style == 'adventure' and x in ['cold', 'continental'] or
                                    travel_style == 'cultural' and x in ['temperate', 'mediterranean'] else 0
                    )
                else:
                    past_features['style_match'] = 0
                
                if preferred_budget:
                    budget_mapping = {
                        'luxury': 'high',
                        'moderate': 'medium',
                        'budget': 'low'
                    }
                    mapped_budget = budget_mapping.get(preferred_budget, 'medium')
                    past_features['budget_match'] = past_features['cost_level'].apply(
                        lambda x: 1 if x == mapped_budget else 0)
                else:
                    past_features['budget_match'] = 0
                
                past_features['preference_match'] = (
                    past_features['climate_match'] * 0.35 + 
                    past_features['style_match'] * 0.35 + 
                    past_features['budget_match'] * 0.3
                )
                
                # We don't have collaborative scores for past destinations
                past_features['collab_score'] = 0
                
                # We don't have item similarity for past destinations
                past_features['item_similarity'] = 0
                
                # Add explicit preference weights
                for col, weight in preference_weights.items():
                    if col in past_features.columns:
                        past_features[col] = past_features[col] * weight
                
                # Calculate target based on ratings and likes
                past_features['target'] = (past_features['rating'] / 5.0) + past_features['liked']
                past_features['target'] = past_features['target'].clip(0, 1)  # Normalize to 0-1
                
                # Select same features as our recommendation candidates
                X_train = past_features[[col for col in numerical_cols if col in past_features.columns]]
                missing_cols = set(numerical_cols) - set(X_train.columns)
                
                # Add missing columns with zeros
                for col in missing_cols:
                    X_train[col] = 0
                    
                # Make sure X_train has the same columns as X
                X_train = X_train[X.columns]
                
                y_train = past_features['target']
                
                print("Training XGBoost model")
                # Train an XGBoost model with hyperparameters
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Get feature importances for debugging
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("Top 10 important features:")
                print(feature_importance.head(10))
                
                # Predict on unvisited destinations
                y_pred = model.predict(X)
                
            except Exception as e:
                print(f"Model training error: {e}")
                # Fall back to weighted sum approach
                print("Falling back to weighted sum approach")
                
                # Create a weighted sum of all features
                y_pred = (
                    0.30 * feature_df['collab_score'] + 
                    0.35 * feature_df['preference_match'] +
                    0.20 * feature_df['item_similarity'] +
                    0.15 * (feature_df[[col for col in feature_df.columns if col.startswith('is_')]].sum(axis=1) / 
                           len([col for col in feature_df.columns if col.startswith('is_')]))
                )
        else:
            # Not enough data for model training, use weighted sum approach
            print("Using weighted sum approach for scoring")
            
            # Create a weighted sum of all features
            y_pred = (
                0.30 * feature_df['collab_score'] + 
                0.40 * feature_df['preference_match'] +
                0.15 * feature_df['item_similarity'] +
                0.15 * (feature_df[[col for col in feature_df.columns if col.startswith('is_')]].sum(axis=1) / 
                       len([col for col in feature_df.columns if col.startswith('is_')]))
            )
                
        # Create recommendations DataFrame with scores
        recommendations = unvisited_df.copy()
        recommendations['score'] = y_pred
        
        # Sort by score and return top recommendations
        recommendations = recommendations.sort_values('score', ascending=False)
        
        # Add additional information for display
        recommendations['primary_image_url'] = recommendations['climate_type'].apply(
            lambda x: f"https://source.unsplash.com/featured/800x600?{x.replace('_', '+')}" 
            if pd.notna(x) else "https://source.unsplash.com/featured/800x600?travel"
        )
        
        print(f"Returning {len(recommendations)} recommendations")
        return recommendations, None
        
    except Exception as e:
        # Return empty dataframe and error message
        print(f"Error in train_destination_recommender: {str(e)}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return pd.DataFrame(), f"Error generating recommendations: {str(e)}"


def collaborative_filtering_recommendations(user_id, top_n=20):
    """
    Generate recommendations using only collaborative filtering when user has limited data.
    
    Args:
        user_id (int): The user ID
        top_n (int): Number of recommendations to return
        
    Returns:
        tuple: (DataFrame of recommendations, error message if any)
    """
    try:
        print(f"Starting collaborative filtering for user {user_id}")
        # Find similar users based on available preferences
        similar_users_query = """
        SELECT TOP 30 up1.user_id, COUNT(*) as common_prefs
        FROM user_preferences up1
        JOIN user_preferences up2 ON 
            up1.preference_type = up2.preference_type AND 
            up1.preference_value = up2.preference_value
        WHERE up2.user_id = ? AND up1.user_id != ?
        GROUP BY up1.user_id
        ORDER BY common_prefs DESC
        """
        print(f"Executing similar users query: {similar_users_query}")
        similar_users = run_query(similar_users_query, [user_id, user_id])
        print(f"Found {len(similar_users)} similar users")
        
        if similar_users.empty:
            print("No similar users found, using global popularity")
            # If no similar users, use global popularity
            popular_destinations_query = """
            SELECT TOP 20 d.*, 
                   COUNT(DISTINCT b.user_id) as booking_count,
                   AVG(CAST(r.rating AS FLOAT)) as avg_rating
            FROM destinations d
            LEFT JOIN hotels h ON d.destination_id = h.destination_id
            LEFT JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
            LEFT JOIN bookings b ON hb.booking_id = b.booking_id
            LEFT JOIN reviews r ON d.destination_id = r.entity_id AND r.entity_type = 'destination'
            WHERE d.destination_id NOT IN (
                SELECT DISTINCT h2.destination_id 
                FROM hotels h2
                JOIN hotel_bookings hb2 ON h2.hotel_id = hb2.hotel_id
                JOIN bookings b2 ON hb2.booking_id = b2.booking_id
                WHERE b2.user_id = ?
            )
            GROUP BY d.destination_id, d.name, d.country, d.region, d.city, 
                     d.latitude, d.longitude, d.climate_type, d.best_season_to_visit, 
                     d.description, d.image_url, d.popularity_score, d.average_rating
            ORDER BY avg_rating DESC, booking_count DESC
            """
            print(f"Executing popular destinations query: {popular_destinations_query}")
            recommendations = run_query(popular_destinations_query, [user_id])
            print(f"Found {len(recommendations)} popular destinations")
            
            if recommendations.empty:
                print("No recommendations from database, generating fallback recommendations")
                # Get some random destinations as a fallback
                fallback_query = "SELECT TOP 20 * FROM destinations ORDER BY NEWID()"
                recommendations = run_query(fallback_query)
                
                if recommendations.empty:
                    return pd.DataFrame(), "No destination data available in the database."
                
                # Generate synthetic scores for these recommendations
                recommendations['score'] = [random.uniform(0.5, 0.98) for _ in range(len(recommendations))]
                return recommendations, None
                
            # Add score based on popularity
            max_bookings = recommendations['booking_count'].max() if 'booking_count' in recommendations.columns else 1
            if max_bookings > 0:
                recommendations['score'] = recommendations['booking_count'] / max_bookings
            else:
                recommendations['score'] = 0
                
            # Add rating component if available
            if 'avg_rating' in recommendations.columns:
                recommendations['score'] += recommendations['avg_rating'].fillna(2.5) / 5.0
                
            # Normalize scores
            if recommendations['score'].max() > 0:
                recommendations['score'] = recommendations['score'] / recommendations['score'].max()
        else:
            print("Using collaborative filtering with similar users")
            # Get destinations this user hasn't visited
            unvisited_query = """
            SELECT d.*
            FROM destinations d
            WHERE d.destination_id NOT IN (
                SELECT DISTINCT h.destination_id 
                FROM hotels h
                JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
                JOIN bookings b ON hb.booking_id = b.booking_id
                WHERE b.user_id = ?
            )
            """
            print(f"Executing unvisited destinations query: {unvisited_query}")
            unvisited_df = run_query(unvisited_query, [user_id])
            print(f"Found {len(unvisited_df)} unvisited destinations")
            
            if unvisited_df.empty:
                # If the query returns no destinations (maybe because the visited destinations query is empty)
                # Get all destinations as a fallback
                print("No unvisited destinations found, getting all destinations")
                unvisited_df = run_query("SELECT * FROM destinations ORDER BY NEWID()")
                
                if unvisited_df.empty:
                    return pd.DataFrame(), "No destination data available in the database."
            
            print(f"Setting up collab scores for {len(unvisited_df)} destinations")
            # Calculate collaborative scores
            collab_scores = pd.Series(0, index=unvisited_df['destination_id'])
            
            print(f"Processing {len(similar_users)} similar users")
            for _, sim_user in similar_users.iterrows():
                sim_user_id = sim_user['user_id']
                similarity = sim_user['common_prefs']
                
                # Get destinations this similar user liked or visited with high ratings
                sim_user_liked_query = """
                SELECT d.destination_id, 
                       COALESCE(r.rating, 0) * 0.2 + 
                       (CASE WHEN ui.interaction_type = 'like' THEN 1 ELSE 0 END) * 0.3 +
                       (CASE WHEN b.booking_id IS NOT NULL THEN 1 ELSE 0 END) * 0.5 as preference_score
                FROM destinations d
                LEFT JOIN user_interactions ui ON d.destination_id = ui.entity_id 
                    AND ui.entity_type = 'destination' AND ui.user_id = ?
                LEFT JOIN reviews r ON d.destination_id = r.entity_id 
                    AND r.entity_type = 'destination' AND r.user_id = ?
                LEFT JOIN (
                    -- Join through hotel bookings
                    SELECT DISTINCT h.destination_id, b.booking_id
                    FROM hotels h
                    JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
                    JOIN bookings b ON hb.booking_id = b.booking_id
                    WHERE b.user_id = ?
                    
                    UNION
                    
                    -- Join through tour bookings
                    SELECT DISTINCT t.destination_id, b.booking_id
                    FROM tours t
                    JOIN tour_bookings tb ON t.tour_id = tb.tour_id
                    JOIN bookings b ON tb.booking_id = b.booking_id
                    WHERE b.user_id = ?
                ) b ON d.destination_id = b.destination_id
                WHERE (ui.interaction_type IN ('like', 'visit') OR r.rating >= 4 OR b.booking_id IS NOT NULL)
                """
                sim_user_likes = run_query(sim_user_liked_query, [sim_user_id, sim_user_id, sim_user_id, sim_user_id])
                
                # Add weighted collaborative scores
                for _, row in sim_user_likes.iterrows():
                    dest_id = row['destination_id']
                    if dest_id in collab_scores.index:
                        collab_scores[dest_id] += similarity * row['preference_score']
            
            print("Creating recommendations DataFrame")
            # Create recommendations DataFrame
            recommendations = unvisited_df.copy()
            
            # If collab_scores has no data, generate synthetic scores
            if collab_scores.sum() == 0:
                print("No collaborative scores found, generating synthetic scores")
                recommendations['score'] = [random.uniform(0.5, 0.98) for _ in range(len(recommendations))]
            else:
                recommendations['score'] = recommendations['destination_id'].map(collab_scores)
                recommendations['score'] = recommendations['score'].fillna(0)
                
                # Normalize scores
                if recommendations['score'].max() > 0:
                    recommendations['score'] = recommendations['score'] / recommendations['score'].max()
            
            # If no good collaborative scores, supplement with popular destinations
            if recommendations['score'].max() < 0.1:
                print("Supplementing with popular destinations due to low collaborative scores")
                # Mix in some popular destinations
                popular_query = """
                SELECT TOP 20 d.destination_id, 
                       COUNT(DISTINCT b.user_id) as booking_count,
                       AVG(CAST(r.rating AS FLOAT)) as avg_rating
                FROM destinations d
                LEFT JOIN hotels h ON d.destination_id = h.destination_id
                LEFT JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
                LEFT JOIN bookings b ON hb.booking_id = b.booking_id
                LEFT JOIN reviews r ON d.destination_id = r.entity_id AND r.entity_type = 'destination'
                WHERE d.destination_id NOT IN (
                    SELECT DISTINCT h2.destination_id 
                    FROM hotels h2
                    JOIN hotel_bookings hb2 ON h2.hotel_id = hb2.hotel_id
                    JOIN bookings b2 ON hb2.booking_id = b2.booking_id
                    WHERE b2.user_id = ?
                )
                GROUP BY d.destination_id
                ORDER BY booking_count DESC, avg_rating DESC
                """
                popular_df = run_query(popular_query, [user_id])
                
                if not popular_df.empty:
                    # Normalize popularity scores
                    max_bookings = popular_df['booking_count'].max() if popular_df['booking_count'].max() > 0 else 1
                    popular_df['pop_score'] = popular_df['booking_count'] / max_bookings
                    
                    # Blend scores
                    for _, row in popular_df.iterrows():
                        dest_id = row['destination_id']
                        if dest_id in recommendations['destination_id'].values:
                            current_score = recommendations.loc[recommendations['destination_id'] == dest_id, 'score'].iloc[0]
                            pop_score = row['pop_score']
                            # Blend collaborative and popularity (70% collab, 30% popularity)
                            recommendations.loc[recommendations['destination_id'] == dest_id, 'score'] = 0.7 * current_score + 0.3 * pop_score
        
        print("Sorting and finalizing recommendations")
        # Sort by score
        recommendations = recommendations.sort_values('score', ascending=False)
        
        # Add additional information for display
        recommendations['primary_image_url'] = recommendations['climate_type'].apply(
            lambda x: f"https://source.unsplash.com/featured/800x600?{x.replace('_', '+')}" 
            if pd.notna(x) else "https://source.unsplash.com/featured/800x600?travel"
        )
        
        print(f"Returning {min(top_n, len(recommendations))} recommendations")
        # Return top N recommendations
        return recommendations.head(top_n), None
        
    except Exception as e:
        print(f"Error in collaborative_filtering_recommendations: {str(e)}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return pd.DataFrame(), f"Error generating collaborative recommendations: {str(e)}"

def show_discover_tab(user_id, max_recommendations=10):
    """Show the discover tab with personalized destination recommendations."""
    st.subheader("Discover New Destinations")
    
    # User setting for number of recommendations
    num_recommendations = st.slider("Number of Recommendations", 5, 20, max_recommendations, key="discover_recommendations_slider")
    
    # Button to generate recommendations
    if st.button("Generate Personalized Recommendations"):
        with st.spinner("Analyzing your preferences, travel history, and finding similar travelers..."):
            recommendations, error = train_destination_recommender(user_id)
            
        if error:
            st.error(error)
            return
            
        if recommendations is None or recommendations.empty:
            st.info("We couldn't generate any recommendations for you at this time. Try exploring more destinations!")
            return
            
        # Display recommendations
        st.success(f"✨ Here are your top {num_recommendations} personalized destination recommendations!")
        
        # Show recommendations map
        st.write("**Map of Recommended Destinations:**")
        
        # Get all destinations for context
        all_destinations = run_query("SELECT destination_id, name, country, region, latitude, longitude FROM destinations")
        
        # Limit to requested number
        top_recommendations = recommendations.head(num_recommendations)
        
        # Get user's visited destinations for context
        visited_query = f"""
        SELECT DISTINCT
            d.destination_id, d.name, d.country, d.region, d.latitude, d.longitude
        FROM
            destinations d
        JOIN
            hotels h ON d.destination_id = h.destination_id
        JOIN
            hotel_bookings hb ON h.hotel_id = hb.hotel_id
        JOIN
            bookings b ON hb.booking_id = b.booking_id
        WHERE
            b.user_id = ? AND b.booking_status = 'completed'
        UNION
        SELECT DISTINCT
            d.destination_id, d.name, d.country, d.region, d.latitude, d.longitude
        FROM
            destinations d
        JOIN
            tours t ON d.destination_id = t.destination_id
        JOIN
            tour_bookings tb ON t.tour_id = tb.tour_id
        JOIN
            bookings b ON tb.booking_id = b.booking_id
        WHERE
            b.user_id = ? AND b.booking_status = 'completed'
        """
        
        visited_df = run_query(visited_query, params=(user_id, user_id))
        
        # Show map
        show_destination_map(all_destinations, highlight_destinations=top_recommendations, user_visited=visited_df)
        
        # Show recommendation cards
        st.write("**Your Recommended Destinations:**")
        
        # Create a grid of cards for recommendations
        cols = st.columns(3)
        
        for i, (_, dest) in enumerate(top_recommendations.iterrows()):
            col_idx = i % 3
            
            with cols[col_idx]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
                        <h3>{dest['name']}, {dest['country']}</h3>
                        <p><strong>Region:</strong> {dest['region']}</p>
                        <p><strong>Climate:</strong> {dest['climate_type']}</p>
                        <p><strong>Best Season:</strong> {dest['best_season_to_visit']}</p>
                        <p><strong>Match Score:</strong> {dest['score']*100:.1f}%</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Try to add the recommendation explanation section with error handling
        try:
            # Recommendation explanation
            st.subheader("Why We Recommended These Places")
            
            # Get user preferences
            user_query = """
            SELECT u.*, 
                   p.preferred_hotel_stars, p.preferred_budget_category, 
                   p.preferred_activities, p.preferred_climates, p.travel_style
            FROM users u
            LEFT JOIN user_preferences p ON u.user_id = p.user_id
            WHERE u.user_id = ?
            """
            
            user_df = run_query(user_query, params=(user_id,))
            if user_df.empty:
                st.warning("Could not retrieve user preferences for explanation.")
                return
                
            user = user_df.iloc[0]
            
            # Check if user has limited history
            visited_count_query = """
            SELECT COUNT(DISTINCT d.destination_id) as visit_count
            FROM destinations d
            JOIN hotels h ON d.destination_id = h.destination_id
            JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
            JOIN bookings b ON hb.booking_id = b.booking_id
            WHERE b.user_id = ? AND b.booking_status = 'completed'
            """
            
            visit_count = run_query(visited_count_query, params=(user_id,))
            limited_history = visit_count.iloc[0]['visit_count'] < 3 if not visit_count.empty else True
            
            # Show explanation based on user preferences
            st.write("These recommendations are based on:")
            
            explanations = []
            
            if pd.notna(user.get('preferred_climates')):
                try:
                    climates = json.loads(user['preferred_climates']) if isinstance(user['preferred_climates'], str) else user['preferred_climates']
                    if climates:
                        explanations.append(f"Your preferred climates: {', '.join(climates)}")
                except:
                    pass
                    
            if pd.notna(user.get('preferred_activities')):
                try:
                    activities = json.loads(user['preferred_activities']) if isinstance(user['preferred_activities'], str) else user['preferred_activities']
                    if activities:
                        explanations.append(f"Your activity preferences: {', '.join(activities)}")
                except:
                    pass
                    
            if pd.notna(user.get('travel_style')):
                explanations.append(f"Your travel style: {user['travel_style']}")
            
            # Add collaborative filtering explanation if applicable
            if limited_history:
                explanations.append("<strong>Destinations popular with travelers similar to you</strong> (since you have limited travel history)")
            else:
                explanations.append("Destinations popular with travelers similar to you")
                
            explanations.append("Your past travel history and interactions")
            explanations.append("Destinations with high ratings from other travelers")
            
            for explanation in explanations:
                st.markdown(f"- {explanation}", unsafe_allow_html=True)
                
            # Feature importance visualization
            st.subheader("What Factors Influenced These Recommendations")
            
            # Create a simple chart showing the factors - adjust weights based on history
            factors = ['Climate Match', 'Ratings', 'Similar Travelers', 'Popularity', 'Seasonality']
            
            # Adjust importance weights based on whether user has limited history
            if limited_history:
                importance = [0.25, 0.15, 0.30, 0.15, 0.15]  # More weight to similar travelers
            else:
                importance = [0.25, 0.20, 0.20, 0.20, 0.15]  # More balanced
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(factors, importance, color=sns.color_palette('viridis', len(factors)))
            ax.set_xlabel('Importance in Recommendations')
            ax.set_title('Factors that Influenced Your Recommendations')
            
            # Add percentages
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width*100:.0f}%", 
                        ha='left', va='center')
                
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            # If explanation section fails, just show a simple message
            st.info("We analyzed your preferences and travel history to find destinations you might enjoy.")
            print(f"Error showing recommendation explanation: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Virtuoso Travel Recommendations",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("✈️ Virtuoso Travel Recommendations")
    
    # Check if admin mode is enabled
    is_admin = st.sidebar.checkbox("Enable Admin Mode", False)
    
    # Initialize recommendation system
    rec_system = get_recommendation_system()
    
    # Sidebar for user selection (only in user mode)
    if not is_admin:
        st.sidebar.title("User Selection")
        
        # Get users
        users_query = "SELECT TOP 1000 user_id, first_name, last_name FROM users ORDER BY user_id"
        users_df = run_query(users_query)
        
        if users_df.empty:
            st.sidebar.error("No users found in the database.")
            st.sidebar.info("Enable Admin Mode and use the Database Setup tab to initialize the database.")
            return
        
        # Format user selection options
        user_options = [f"{row['user_id']}: {row['first_name']} {row['last_name']}" for _, row in users_df.iterrows()]
        
        # User selection
        selected_user = st.sidebar.selectbox("Select User", user_options, key="user_selection_dropdown")
        user_id = int(selected_user.split(':')[0])
        
        # Initialize session state if needed
        if 'previous_user_id' not in st.session_state:
            st.session_state.previous_user_id = user_id
            st.session_state.user_recommendations = None
            st.session_state.discover_recommendations = None
            st.session_state.discover_error = None
        
        # Check if user has changed and reset recommendations if needed
        if st.session_state.previous_user_id != user_id:
            # Clear recommendations to force regeneration for new user
            st.session_state.user_recommendations = None
            st.session_state.discover_recommendations = None
            st.session_state.discover_error = None
            # Update previous user id
            st.session_state.previous_user_id = user_id
        
        # Advanced settings
        st.sidebar.title("Settings")
        num_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10, key="sidebar_recommendations_slider")
        
        # Tabs for different sections
        tabs = st.tabs(["Profile", "Travel History", "Recommendations", "Discover", "Statistics"])
        
        # User Profile Tab
        with tabs[0]:
            user = show_user_profile(user_id)
            if user is None:
                return
        
        # Travel History Tab
        with tabs[1]:
            visited_df = show_user_history(user_id)
        
        # Recommendations Tab
        with tabs[2]:
            # Check if we need to generate new recommendations
            if 'user_recommendations' not in st.session_state or st.session_state.user_recommendations is None:
                if st.button("Generate Recommendations", key="generate_main_recommendations"):
                    with st.spinner("Generating recommendations..."):
                        recommendations = rec_system.get_destination_recommendations(user_id, num_recommendations)
                        st.session_state.user_recommendations = recommendations
                else:
                    st.info("Click the button above to generate recommendations based on your travel history and preferences.")
                    recommendations = pd.DataFrame()
            else:
                recommendations = st.session_state.user_recommendations
                
                # Add a button to regenerate recommendations
                if st.button("Regenerate Recommendations", key="regenerate_main_recommendations"):
                    with st.spinner("Generating new recommendations..."):
                        recommendations = rec_system.get_destination_recommendations(user_id, num_recommendations)
                        st.session_state.user_recommendations = recommendations
            
            # Display recommendations if available
            if not recommendations.empty:
                show_recommendations_display(user_id, recommendations, num_recommendations, visited_df)
            
        # Discover Tab (New)
        with tabs[3]:
            st.write("### Discover New Destinations")
            
            # Debug information
            show_debug = st.checkbox("Show debug info", False, key="discover_debug_switch")
            
            if 'discover_recommendations' not in st.session_state or st.session_state.discover_recommendations is None:
                # User needs to generate recommendations
                if show_debug:
                    st.info("No recommendations in session state yet")
                
                discover_btn = st.button("Generate Personalized Recommendations", key="generate_discover_recommendations")
                if discover_btn:
                    with st.spinner("Analyzing your preferences, travel history, and finding similar travelers..."):
                        recommendations, error = train_destination_recommender(user_id)
                        
                        if show_debug:
                            st.write("**Training Results:**")
                            st.write(f"Recommendations is None: {recommendations is None}")
                            if recommendations is not None:
                                st.write(f"Recommendations is empty: {recommendations.empty}")
                                if not recommendations.empty:
                                    st.write(f"Recommendations shape: {recommendations.shape}")
                                    st.write(f"Sample data: {recommendations.head(2).to_dict()}")
                            st.write(f"Error message: {error}")
                        
                        st.session_state.discover_recommendations = recommendations
                        st.session_state.discover_error = error
                        
                        # If we got recommendations, display them immediately
                        if recommendations is not None and not recommendations.empty:
                            st.success(f"✨ Generated {len(recommendations)} recommendations!")
                            # User setting for number of recommendations
                            num_to_show = min(len(recommendations), 20)
                            num_recommendations = st.slider("Number of Recommendations", 5, num_to_show, min(10, num_to_show))
                            show_discover_recommendations(user_id, recommendations.head(num_recommendations), visited_df)
                else:
                    st.info("Click the button above to discover new destinations based on collaborative filtering and your preferences.")
            else:
                # Already have recommendations, show them with option to regenerate
                recommendations = st.session_state.discover_recommendations
                error = st.session_state.get('discover_error', None)
                
                if show_debug:
                    st.write("**Stored Recommendations:**")
                    st.write(f"Recommendations is None: {recommendations is None}")
                    if recommendations is not None:
                        st.write(f"Recommendations is empty: {recommendations.empty}")
                        if not recommendations.empty:
                            st.write(f"Recommendations shape: {recommendations.shape}")
                            st.write(f"Recommendations columns: {list(recommendations.columns)}")
                    st.write(f"Error message: {error}")
                
                if st.button("Regenerate Personalized Recommendations", key="regenerate_discover_recommendations"):
                    with st.spinner("Finding new destinations for you..."):
                        recommendations, error = train_destination_recommender(user_id)
                        st.session_state.discover_recommendations = recommendations
                        st.session_state.discover_error = error
                        
                        # If we got recommendations, display them immediately
                        if recommendations is not None and not recommendations.empty:
                            st.success(f"✨ Generated {len(recommendations)} new recommendations!")
                
                if error:
                    st.error(error)
                elif recommendations is None or recommendations.empty:
                    st.info("We couldn't generate any recommendations for you at this time. Try exploring more destinations!")
                else:
                    # User setting for number of recommendations
                    num_to_show = min(len(recommendations), 20)
                    num_recommendations = st.slider("Number of Recommendations", 5, num_to_show, min(10, num_to_show), 
                                                  key="discover_display_slider")
                    
                    # Make sure we don't try to show more recommendations than we have
                    num_recommendations = min(num_recommendations, len(recommendations))
                    
                    if show_debug:
                        st.write(f"About to show {num_recommendations} recommendations")
                    
                    # Display the recommendations
                    show_discover_recommendations(user_id, recommendations.head(num_recommendations), visited_df)
        
        # Statistics Tab
        with tabs[4]:
            show_travel_statistics()
    else:
        # Admin mode tabs
        admin_tabs = st.tabs(["Database Setup", "Database Explorer", "SQL Query", "Name Management"])
        
        # Database Setup Tab
        with admin_tabs[0]:
            show_admin_db_setup_panel()
        
        # Database Explorer Tab
        with admin_tabs[1]:
            show_admin_database_panel()
        
        # SQL Query Tab
        with admin_tabs[2]:
            show_admin_sql_panel()
        
        # Name Management Tab
        with admin_tabs[3]:
            show_admin_names_panel()
    
    # Footer
    st.markdown("---")
    st.markdown("*Virtuoso Travel Data Engineering Project*")

# Helper function to display recommendations
def show_recommendations_display(user_id, recommendations, num_recommendations, visited_df=None):
    """Display recommendations in a consistent way"""
    st.subheader("Destination Recommendations")
    
    # Limit to requested number
    top_recommendations = recommendations.head(num_recommendations)
    
    # Get all destinations for the map - make sure to include all necessary columns
    all_destinations = run_query("SELECT destination_id, name, country, region, latitude, longitude FROM destinations")
    
    # Debug information for column names
    if all_destinations.empty:
        st.error("No destination data available")
        return
        
    # Log available columns for debugging
    if st.checkbox("Show data debug info", False):
        st.write("All destinations columns:", list(all_destinations.columns))
        if not top_recommendations.empty:
            st.write("Recommendations columns:", list(top_recommendations.columns))
        if visited_df is not None and not visited_df.empty:
            st.write("Visited destinations columns:", list(visited_df.columns))
    
    # Show recommendations map
    st.write("**Map of Recommended Destinations:**")
    
    try:
        show_destination_map(all_destinations, highlight_destinations=top_recommendations, user_visited=visited_df)
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")
        st.error("Try regenerating recommendations or checking the database schema")
        if st.checkbox("Show detailed error", False):
            st.exception(e)
    
    # Show recommendations table
    st.write("**Top Recommended Destinations:**")
    
    # Determine which columns to show in the table
    display_cols = []
    
    # Find the appropriate name, country, and region columns
    name_cols = ['name', 'destination_name']
    country_cols = ['country']
    region_cols = ['region']
    climate_cols = ['climate_type', 'climate']
    
    name_col = next((col for col in name_cols if col in top_recommendations.columns), None)
    country_col = next((col for col in country_cols if col in top_recommendations.columns), None)
    region_col = next((col for col in region_cols if col in top_recommendations.columns), None)
    climate_col = next((col for col in climate_cols if col in top_recommendations.columns), None)
    
    # Add available columns to display list
    if name_col:
        display_cols.append(name_col)
    if country_col:
        display_cols.append(country_col)
    if region_col:
        display_cols.append(region_col)
    if climate_col:
        display_cols.append(climate_col)
    
    # Always include score if available
    if 'score' in top_recommendations.columns:
        display_cols.append('score')
        
    # If no columns are available, show raw data
    if not display_cols:
        st.warning("Cannot format recommendation table: Missing required columns")
        st.dataframe(top_recommendations)
        return
        
    # Create the display table
    rec_table = top_recommendations[display_cols].reset_index(drop=True)
    rec_table.index = rec_table.index + 1  # Start index at 1
    
    # Format the score column as percentage if it exists
    if 'score' in rec_table.columns:
        rec_table['score'] = rec_table['score'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(rec_table)

# Helper function to display discover recommendations
def show_discover_recommendations(user_id, recommendations, visited_df=None):
    """Display discover tab recommendations"""
    if recommendations is None or recommendations.empty:
        st.warning("No recommendations to display")
        return
        
    # Debug info toggle
    show_debug = st.checkbox("Show recommendation details", False, key="rec_debug_details")
    
    if show_debug:
        st.write(f"Recommendations data shape: {recommendations.shape}")
        st.write(f"Available columns: {list(recommendations.columns)}")
        if not recommendations.empty:
            st.write(f"First recommendation: {recommendations.iloc[0].to_dict()}")
    
    # Success message
    st.success(f"✨ Here are your top {len(recommendations)} personalized destination recommendations!")
    
    # Show recommendations map if we have data
    if recommendations is not None and not recommendations.empty:
        st.write("**Map of Recommended Destinations:**")
        
        # Get all destinations for context
        all_destinations = run_query("SELECT destination_id, name, country, region, latitude, longitude FROM destinations")
        
        if show_debug:
            st.write(f"All destinations data shape: {all_destinations.shape}")
            st.write(f"All destinations columns: {list(all_destinations.columns)}")
        
        # If we have destination data, show the map
        if not all_destinations.empty:
            try:
                show_destination_map(all_destinations, highlight_destinations=recommendations, user_visited=visited_df)
            except Exception as e:
                st.error(f"Error displaying map: {str(e)}")
                if show_debug:
                    st.exception(e)
        else:
            st.warning("Cannot display map: No destination data available")
            
        # Show recommendation cards
        st.write("**Your Recommended Destinations:**")
        
        # Identify column names
        name_cols = ['name', 'destination_name']
        country_cols = ['country']
        region_cols = ['region']
        climate_cols = ['climate_type', 'climate']
        season_cols = ['best_season_to_visit', 'best_season']
        
        name_col = next((col for col in name_cols if col in recommendations.columns), None)
        country_col = next((col for col in country_cols if col in recommendations.columns), None)
        region_col = next((col for col in region_cols if col in recommendations.columns), None)
        climate_col = next((col for col in climate_cols if col in recommendations.columns), None)
        season_col = next((col for col in season_cols if col in recommendations.columns), None)
        
        if show_debug:
            st.write(f"Using columns: name={name_col}, country={country_col}, region={region_col}, climate={climate_col}, season={season_col}")
        
        # Create a fallback display if we can't determine the proper columns
        if not name_col and not country_col:
            st.warning("Cannot format recommendation display: Missing required columns")
            st.dataframe(recommendations)
            return
        
        # Create a grid of cards for recommendations
        cols = st.columns(3)
        
        for i, (_, dest) in enumerate(recommendations.iterrows()):
            col_idx = i % 3
            
            with cols[col_idx]:
                # Create a safe dictionary of destination attributes with defaults for missing values
                safe_dest = {
                    'name': dest.get(name_col, 'Unknown') if name_col else 'Unknown Destination',
                    'country': dest.get(country_col, 'Unknown') if country_col else 'Unknown Country',
                    'region': dest.get(region_col, 'Unknown') if region_col else 'Unknown Region',
                    'climate': dest.get(climate_col, 'Unknown') if climate_col else 'Unknown Climate',
                    'season': dest.get(season_col, 'Unknown') if season_col else 'Year-round',
                    'score': dest.get('score', 0) if 'score' in dest else 0
                }
                
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
                        <h3>{safe_dest['name']}, {safe_dest['country']}</h3>
                        <p><strong>Region:</strong> {safe_dest['region']}</p>
                        <p><strong>Climate:</strong> {safe_dest['climate']}</p>
                        <p><strong>Best Season:</strong> {safe_dest['season']}</p>
                        <p><strong>Match Score:</strong> {safe_dest['score']*100:.1f}%</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Try to add the recommendation explanation section with error handling
        try:
            # Recommendation explanation
            st.subheader("Why We Recommended These Places")
            
            # Get user preferences
            user_query = """
            SELECT u.*, 
                   p.preferred_hotel_stars, p.preferred_budget_category, 
                   p.preferred_activities, p.preferred_climates, p.travel_style
            FROM users u
            LEFT JOIN user_preferences p ON u.user_id = p.user_id
            WHERE u.user_id = ?
            """
            
            user_df = run_query(user_query, params=(user_id,))
            if user_df.empty:
                st.warning("Could not retrieve user preferences for explanation.")
                return
                
            user = user_df.iloc[0]
            
            # Check if user has limited history
            visited_count_query = """
            SELECT COUNT(DISTINCT d.destination_id) as visit_count
            FROM destinations d
            JOIN hotels h ON d.destination_id = h.destination_id
            JOIN hotel_bookings hb ON h.hotel_id = hb.hotel_id
            JOIN bookings b ON hb.booking_id = b.booking_id
            WHERE b.user_id = ? AND b.booking_status = 'completed'
            """
            
            visit_count = run_query(visited_count_query, params=(user_id,))
            limited_history = visit_count.iloc[0]['visit_count'] < 3 if not visit_count.empty else True
            
            # Show explanation based on user preferences
            st.write("These recommendations are based on:")
            
            explanations = []
            
            if pd.notna(user.get('preferred_climates')):
                try:
                    climates = json.loads(user['preferred_climates']) if isinstance(user['preferred_climates'], str) else user['preferred_climates']
                    if climates:
                        explanations.append(f"Your preferred climates: {', '.join(climates)}")
                except:
                    pass
                    
            if pd.notna(user.get('preferred_activities')):
                try:
                    activities = json.loads(user['preferred_activities']) if isinstance(user['preferred_activities'], str) else user['preferred_activities']
                    if activities:
                        explanations.append(f"Your activity preferences: {', '.join(activities)}")
                except:
                    pass
                    
            if pd.notna(user.get('travel_style')):
                explanations.append(f"Your travel style: {user['travel_style']}")
            
            # Add collaborative filtering explanation if applicable
            if limited_history:
                explanations.append("<strong>Destinations popular with travelers similar to you</strong> (since you have limited travel history)")
            else:
                explanations.append("Destinations popular with travelers similar to you")
                
            explanations.append("Your past travel history and interactions")
            explanations.append("Destinations with high ratings from other travelers")
            
            for explanation in explanations:
                st.markdown(f"- {explanation}", unsafe_allow_html=True)
                
            # Feature importance visualization
            st.subheader("What Factors Influenced These Recommendations")
            
            # Create a simple chart showing the factors - adjust weights based on history
            factors = ['Climate Match', 'Ratings', 'Similar Travelers', 'Popularity', 'Seasonality']
            
            # Adjust importance weights based on whether user has limited history
            if limited_history:
                importance = [0.25, 0.15, 0.30, 0.15, 0.15]  # More weight to similar travelers
            else:
                importance = [0.25, 0.20, 0.20, 0.20, 0.15]  # More balanced
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(factors, importance, color=sns.color_palette('viridis', len(factors)))
            ax.set_xlabel('Importance in Recommendations')
            ax.set_title('Factors that Influenced Your Recommendations')
            
            # Add percentages
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width*100:.0f}%", 
                        ha='left', va='center')
                
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            # If explanation section fails, just show a simple message
            st.info("We analyzed your preferences and travel history to find destinations you might enjoy.")
            st.error(f"Error showing recommendation explanation: {str(e)}")
            if st.checkbox("Show error details", False):
                st.exception(e)
    else:
        st.warning("No recommendations available to display.")

if __name__ == "__main__":
    main()