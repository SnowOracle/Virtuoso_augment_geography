import os
import pandas as pd
import pyodbc
import time

def create_database(database_name="virtuoso_travel"):
    """
    Create a new SQL Server database and populate it with our API data.
    
    Args:
        database_name: Name of the database to create
        
    Returns:
        Connection string for the new database
    """
    # Check if API data directory exists
    if not os.path.exists('api_data'):
        # Try to use regular data directory if API data doesn't exist
        if os.path.exists('data'):
            print("API data directory not found. Using regular data directory.")
            data_dir = 'data'
        else:
            raise FileNotFoundError("Data directory not found. Run api_data_fetcher.py or data_generator.py first.")
    else:
        data_dir = 'api_data'

    # SQL Server connection parameters
    server = "127.0.0.1"
    port = "1433"
    uid = "climbing_user"
    pwd = "hoosierheights"
    
    # Create a connection to master database to create our new database
    master_conn_str = (
        f"DRIVER=/opt/homebrew/lib/libtdsodbc.so;"
        f"SERVER={server};"
        f"PORT={port};"
        f"DATABASE=master;"
        f"UID={uid};"
        f"PWD={pwd};"
        f"TDS_Version=7.4;"
    )
    
    try:
        # Connect to master database
        conn = pyodbc.connect(master_conn_str)
        cursor = conn.cursor()
        
        # Check if database already exists and drop it if it does
        print(f"Checking if database '{database_name}' already exists...")
        cursor.execute(f"SELECT database_id FROM sys.databases WHERE name = '{database_name}'")
        if cursor.fetchone():
            print(f"Database '{database_name}' already exists. Dropping it...")
            # Terminate existing connections to the database
            cursor.execute(f"""
            ALTER DATABASE [{database_name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE
            """)
            # Drop the database
            cursor.execute(f"DROP DATABASE [{database_name}]")
            conn.commit()
            print(f"Database '{database_name}' dropped successfully.")
        
        # Create the new database
        print(f"Creating database '{database_name}'...")
        cursor.execute(f"CREATE DATABASE [{database_name}]")
        conn.commit()
        
        # Close the connection to master
        cursor.close()
        conn.close()
        
        print(f"Database '{database_name}' created successfully.")
        
    except pyodbc.Error as e:
        print(f"Error connecting to SQL Server or creating database: {e}")
        raise
    
    # Create a connection string for the new database
    conn_str = (
        f"DRIVER=/opt/homebrew/lib/libtdsodbc.so;"
        f"SERVER={server};"
        f"PORT={port};"
        f"DATABASE={database_name};"
        f"UID={uid};"
        f"PWD={pwd};"
        f"TDS_Version=7.4;"
    )
    
    # Create tables in the new database
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # Define tables
    table_schemas = {
        'users': """
            CREATE TABLE users (
                user_id INT PRIMARY KEY,
                first_name NVARCHAR(100),
                last_name NVARCHAR(100),
                email NVARCHAR(255),
                date_of_birth DATE,
                signup_date DATE,
                preferred_language NVARCHAR(50),
                country_of_residence NVARCHAR(100),
                loyalty_tier NVARCHAR(50)
            )
        """,
        
        'destinations': """
            CREATE TABLE destinations (
                destination_id INT PRIMARY KEY,
                name NVARCHAR(255),
                country NVARCHAR(100),
                region NVARCHAR(100),
                city NVARCHAR(100),
                latitude FLOAT,
                longitude FLOAT,
                popularity_score FLOAT,
                average_rating FLOAT,
                climate_type NVARCHAR(50),
                best_season_to_visit NVARCHAR(50),
                description NVARCHAR(MAX),
                place_type NVARCHAR(100),
                image_url NVARCHAR(1000)
            )
        """,
        
        'hotels': """
            CREATE TABLE hotels (
                hotel_id INT PRIMARY KEY,
                name NVARCHAR(255),
                destination_id INT,
                star_rating INT,
                price_category NVARCHAR(10),
                has_pool BIT,
                has_spa BIT,
                has_restaurant BIT,
                rooms_available INT,
                average_rating FLOAT,
                address NVARCHAR(255),
                latitude FLOAT,
                longitude FLOAT,
                image_url NVARCHAR(1000),
                FOREIGN KEY (destination_id) REFERENCES destinations(destination_id)
            )
        """,
        
        'user_preferences': """
            CREATE TABLE user_preferences (
                preference_id INT PRIMARY KEY,
                user_id INT,
                preferred_hotel_stars INT,
                preferred_budget_category NVARCHAR(50),
                preferred_activities NVARCHAR(MAX),
                preferred_climates NVARCHAR(MAX),
                travel_style NVARCHAR(50),
                maximum_flight_duration INT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """,
        
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
        
        'tours': """
            CREATE TABLE tours (
                tour_id INT PRIMARY KEY,
                name NVARCHAR(255),
                destination_id INT,
                duration_hours INT,
                price FLOAT,
                category NVARCHAR(50),
                group_size_limit INT,
                difficulty_level NVARCHAR(50),
                average_rating FLOAT,
                description NVARCHAR(MAX),
                availability_schedule NVARCHAR(50),
                FOREIGN KEY (destination_id) REFERENCES destinations(destination_id)
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
    
    # Create all tables
    print("Creating tables...")
    try:
        cursor.execute("BEGIN TRANSACTION")
        
        for table_name, schema in table_schemas.items():
            print(f"Creating table: {table_name}")
            cursor.execute(schema)
            
        conn.commit()
        
        # Verify tables were created
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'")
        tables = cursor.fetchall()
        print("Tables in database:", [t[0] for t in tables])
    except Exception as e:
        conn.rollback()
        print(f"Error creating tables: {e}")
        raise
    
    # Load data from CSV files
    print("Loading data from CSV files...")
    csv_files = [
        ('users', f'{data_dir}/users.csv'),
        ('destinations', f'{data_dir}/destinations.csv'),
        ('hotels', f'{data_dir}/hotels.csv'),
        ('tours', f'{data_dir}/tours.csv'),
        ('bookings', f'{data_dir}/bookings.csv'),
        ('hotel_bookings', f'{data_dir}/hotel_bookings.csv'),
        ('tour_bookings', f'{data_dir}/tour_bookings.csv'),
        ('reviews', f'{data_dir}/reviews.csv'),
        ('user_interactions', f'{data_dir}/user_interactions.csv'),
    ]
    
    for table_name, file_path in csv_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            if table_name in ['tours'] and data_dir == 'api_data':
                # Generate synthetic tours data since we don't have it from the API
                generate_synthetic_tours(conn, cursor)
            continue
        
        print(f"Loading data into {table_name} from {file_path}...")
        
        try:
            # Read CSV into DataFrame
            df = pd.read_csv(file_path)
            
            # For SQL Server, boolean columns need to be explicitly converted
            if table_name == 'hotels':
                if 'has_pool' in df.columns:
                    df['has_pool'] = df['has_pool'].astype(int)
                if 'has_spa' in df.columns:
                    df['has_spa'] = df['has_spa'].astype(int)
                if 'has_restaurant' in df.columns:
                    df['has_restaurant'] = df['has_restaurant'].astype(int)
            
            # Check for existing primary keys to avoid duplication
            if table_name == 'users':
                # Get existing user IDs
                cursor.execute("SELECT user_id FROM users")
                existing_ids = set([row[0] for row in cursor.fetchall()])
                
                # Remove rows with IDs that already exist
                if existing_ids:
                    print(f"Checking for duplicate user IDs. Found {len(existing_ids)} existing users.")
                    df = df[~df['user_id'].isin(existing_ids)]
                    print(f"After removing duplicates, {len(df)} users remain to be inserted.")
                    if len(df) == 0:
                        print("No new users to insert. Skipping.")
                        continue
            
            # Batch insert records for performance
            batch_size = 1000
            total_rows = len(df)
            batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
            
            print(f"Inserting {total_rows} rows in {batches} batches...")
            
            for batch in range(batches):
                try:
                    cursor.execute("BEGIN TRANSACTION")
                    
                    start_idx = batch * batch_size
                    end_idx = min((batch + 1) * batch_size, total_rows)
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    # Get column names for this table
                    cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'")
                    db_columns = [row[0].lower() for row in cursor.fetchall()]
                    
                    # Filter DataFrame to only include columns that exist in the table
                    valid_columns = [col for col in batch_df.columns if col.lower() in db_columns]
                    batch_df = batch_df[valid_columns]
                    
                    # Prepare parameter placeholders and column list
                    placeholders = ','.join(['?' for _ in valid_columns])
                    columns = ','.join(valid_columns)
                    
                    # Construct the insert query
                    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                    
                    # Execute batch insert
                    rows_to_insert = [tuple(row) for row in batch_df.values]
                    cursor.executemany(insert_query, rows_to_insert)
                    conn.commit()
                    
                    print(f"  Batch {batch+1}/{batches} inserted ({end_idx-start_idx} rows)")
                except Exception as e:
                    conn.rollback()
                    print(f"Error inserting batch {batch+1} into {table_name}: {e}")
                    # Continue with next batch despite the error
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Continue with next file despite the error
    
    if data_dir == 'api_data':
        # Generate missing transaction data for API data
        print("Generating transaction data...")
        generate_synthetic_transaction_data(conn, cursor)
    
    # Create indices for better performance
    print("Creating indices...")
    
    indices = [
        "CREATE INDEX idx_users_id ON users(user_id)",
        "CREATE INDEX idx_destinations_id ON destinations(destination_id)",
        "CREATE INDEX idx_destinations_country ON destinations(country)",
        "CREATE INDEX idx_destinations_region ON destinations(region)",
        "CREATE INDEX idx_hotels_id ON hotels(hotel_id)",
        "CREATE INDEX idx_hotels_destination ON hotels(destination_id)",
        "CREATE INDEX idx_tours_id ON tours(tour_id)",
        "CREATE INDEX idx_tours_destination ON tours(destination_id)",
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
            # Check if index already exists before creating
            index_name = index_query.split("CREATE INDEX ")[1].split(" ON ")[0]
            cursor.execute(f"SELECT 1 FROM sys.indexes WHERE name = '{index_name}'")
            if cursor.fetchone():
                print(f"Index {index_name} already exists, skipping...")
                continue
            
            # Create the index in its own transaction
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute(index_query)
            conn.commit()
        except pyodbc.Error as e:
            # Skip if index already exists
            print(f"Error creating index: {e}")
            # Rollback the transaction if it failed
            conn.rollback()
    
    print("Indices created successfully")
    
    # Close connection
    cursor.close()
    conn.close()
    print("Database setup complete")
    
    return conn_str

def generate_synthetic_tours(conn, cursor):
    """
    Generate synthetic tours data
    
    Args:
        conn: Database connection
        cursor: Database cursor
    """
    import random
    import numpy as np
    from datetime import datetime
    
    print("Generating synthetic tours data...")
    
    # Get destinations to associate tours with
    cursor.execute("SELECT destination_id, name, country FROM destinations")
    destinations = cursor.fetchall()
    
    if not destinations:
        print("No destinations found. Skipping tour generation.")
        return
    
    # Define tour types and difficulty levels
    tour_types = ['Adventure', 'Cultural', 'Culinary', 'Historical', 'Nature', 
                 'Relaxation', 'Religious', 'Wildlife', 'Photography', 'Arts']
    difficulty_levels = ['Easy', 'Moderate', 'Challenging', 'Difficult', 'Extreme']
    availability_options = ['Daily', 'Weekdays', 'Weekends', 'Seasonal']
    
    # Generate tours
    tours = []
    num_tours = 800  # Target number of tours to generate
    batch_size = 100  # Process tours in batches
    
    for batch_start in range(1, num_tours + 1, batch_size):
        # Start a transaction for this batch of tours
        cursor.execute("BEGIN TRANSACTION")
        try:
            batch_end = min(batch_start + batch_size - 1, num_tours)
            
            for tour_id in range(batch_start, batch_end + 1):
                # Pick a random destination
                destination = random.choice(destinations)
                destination_id = destination[0]
                destination_name = destination[1]
                tour_type = random.choice(tour_types)
                
                price = round(random.uniform(20, 2000), 2)
                duration = random.randint(1, 72)  # Hours
                group_size = random.randint(4, 50)
                difficulty = random.choice(difficulty_levels)
                rating = round(random.uniform(3.0, 5.0), 1)
                
                tour_name = f"{destination_name} {tour_type} Experience"
                description = f"Enjoy this amazing {tour_type.lower()} tour in {destination_name}!"
                
                # Insert tour into database
                cursor.execute("""
                    INSERT INTO tours 
                        (tour_id, name, destination_id, duration_hours, price, category, 
                        group_size_limit, difficulty_level, average_rating, description, availability_schedule)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, 
                    (tour_id, tour_name, destination_id, duration, price, tour_type, 
                    group_size, difficulty, rating, description, random.choice(availability_options))
                )
            
            # Commit the batch transaction
            conn.commit()
            print(f"  Generated tours {batch_start} to {batch_end}")
            
        except Exception as e:
            # Rollback the transaction if there was an error
            conn.rollback()
            print(f"Error generating tours batch {batch_start}-{batch_end}: {e}")
    
    print(f"Generated {num_tours} synthetic tours.")

def generate_synthetic_transaction_data(conn, cursor):
    """
    Generate synthetic transaction data (bookings, reviews, interactions)
    
    Args:
        conn: Database connection
        cursor: Database cursor
    """
    import random
    import numpy as np
    from datetime import datetime, timedelta
    import json
    
    print("Generating synthetic transaction data...")
    
    # Get users to associate bookings with
    cursor.execute("SELECT user_id FROM users")
    user_ids = [row[0] for row in cursor.fetchall()]
    
    if not user_ids:
        print("No users found. Skipping transaction data generation.")
        return
    
    # Get hotels to associate bookings with
    cursor.execute("SELECT hotel_id, destination_id FROM hotels")
    hotels = cursor.fetchall()
    
    if not hotels:
        print("No hotels found. Skipping hotel booking generation.")
        return
    
    # Get tours to associate bookings with
    cursor.execute("SELECT tour_id, destination_id, price FROM tours")
    tours = cursor.fetchall()
    
    if not tours:
        print("No tours found. Skipping tour booking generation.")
    
    # Get destinations for interactions
    cursor.execute("SELECT destination_id FROM destinations")
    destination_ids = [row[0] for row in cursor.fetchall()]
    
    # Generate bookings
    print("Generating bookings...")
    booking_channels = ['web', 'mobile', 'agent']
    booking_statuses = ['confirmed', 'canceled', 'completed']
    room_types = ['Standard', 'Deluxe', 'Suite', 'Family', 'Executive']
    
    num_bookings = min(3000, len(user_ids) * 3)  # Max 3 bookings per user on average
    bookings = []
    hotel_bookings = []
    tour_bookings = []
    reviews = []
    
    # Process bookings in smaller batches to avoid large transactions
    batch_size = 50
    num_batches = (num_bookings + batch_size - 1) // batch_size
    
    for batch in range(num_batches):
        # Start a transaction for this batch
        cursor.execute("BEGIN TRANSACTION")
        try:
            start_idx = batch * batch_size + 1
            end_idx = min((batch + 1) * batch_size, num_bookings)
            
            for booking_id in range(start_idx, end_idx + 1):
                # Pick a random user
                user_id = random.choice(user_ids)
                
                # Generate a booking date in the past 2 years
                booking_days_ago = random.randint(1, 2*365)
                booking_date = (datetime.now() - timedelta(days=booking_days_ago)).strftime('%Y-%m-%d')
                
                # Trip generally happens after booking (could be up to a year later)
                trip_days_after_booking = random.randint(7, 365)
                trip_start_date = (datetime.now() - timedelta(days=booking_days_ago) + timedelta(days=trip_days_after_booking)).strftime('%Y-%m-%d')
                
                # Generate a status weighted toward completed for older bookings
                if booking_days_ago > 365:
                    status_weights = [0.05, 0.15, 0.8]  # mostly completed
                elif booking_days_ago > 180:
                    status_weights = [0.1, 0.2, 0.7]  # many completed
                elif booking_days_ago > 90:
                    status_weights = [0.2, 0.2, 0.6]  # some completed
                elif booking_days_ago > 30:
                    status_weights = [0.6, 0.3, 0.1]  # mostly confirmed
                else:
                    status_weights = [0.8, 0.15, 0.05]  # almost all confirmed
                
                status = np.random.choice(booking_statuses, p=status_weights)
                
                # Generate booking
                total_cost = 0  # Will be calculated based on hotel and tour bookings
                
                # Insert booking
                cursor.execute("""
                    INSERT INTO bookings 
                        (booking_id, user_id, booking_date, total_cost, payment_status, booking_status, booking_channel)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, 
                    (booking_id, user_id, booking_date, total_cost, 
                    'paid' if status != 'canceled' else 'refunded', 
                    status, random.choice(booking_channels))
                )
                
                # Randomly decide if this booking includes a hotel
                if random.random() < 0.9:  # 90% of bookings include hotel
                    hotel = random.choice(hotels)
                    hotel_id = hotel[0]
                    destination_id = hotel[1]
                    
                    stay_length = random.randint(1, 14)  # 1-14 nights
                    
                    check_in_date = trip_start_date
                    check_out_date = (datetime.strptime(check_in_date, '%Y-%m-%d') + timedelta(days=stay_length)).strftime('%Y-%m-%d')
                    
                    # Generate rate based on price category
                    cursor.execute("SELECT price_category FROM hotels WHERE hotel_id = ?", (hotel_id,))
                    price_category = cursor.fetchone()[0]
                    rate_per_night = float(price_category.count('$')) * 100 + random.uniform(0, 100)
                    
                    hotel_cost = rate_per_night * stay_length
                    total_cost += hotel_cost
                    
                    hotel_booking_id = len(hotel_bookings) + 1
                    
                    # Insert hotel booking
                    cursor.execute("""
                        INSERT INTO hotel_bookings 
                            (hotel_booking_id, booking_id, hotel_id, check_in_date, check_out_date, 
                            room_type, number_of_guests, special_requests, rate_per_night)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, 
                        (hotel_booking_id, booking_id, hotel_id, check_in_date, check_out_date,
                        random.choice(room_types), random.randint(1, 6),
                        "None" if random.random() < 0.7 else "Late check-out requested",
                        round(rate_per_night, 2))
                    )
                    
                    # Generate review for hotel if booking is completed
                    if status == 'completed' and random.random() < 0.7:  # 70% chance of leaving a review
                        # Review happens after check-out
                        days_after_checkout = random.randint(1, 30)
                        review_date = (datetime.strptime(check_out_date, '%Y-%m-%d') + timedelta(days=days_after_checkout)).strftime('%Y-%m-%d')
                        
                        # Rating tends to be high but can be low
                        if random.random() < 0.8:  # 80% happy customers
                            rating = random.randint(4, 5)
                            
                            # Get hotel name
                            cursor.execute("SELECT name FROM hotels WHERE hotel_id = ?", (hotel_id,))
                            hotel_name = cursor.fetchone()[0]
                            
                            comment = f"Great stay at {hotel_name}! Would recommend."
                        else:
                            rating = random.randint(1, 3)
                            
                            # Get hotel name
                            cursor.execute("SELECT name FROM hotels WHERE hotel_id = ?", (hotel_id,))
                            hotel_name = cursor.fetchone()[0]
                            
                            comment = f"Disappointing experience at {hotel_name}."
                        
                        review_id = len(reviews) + 1
                        
                        # Insert review
                        cursor.execute("""
                            INSERT INTO reviews 
                                (review_id, user_id, entity_type, entity_id, rating, comment, review_date, helpful_votes)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, 
                            (review_id, user_id, 'hotel', hotel_id, rating, comment, review_date, random.randint(0, 50))
                        )
                        
                        reviews.append(review_id)
                
                # Randomly decide if this booking includes tours
                num_tours_booked = np.random.choice([0, 1, 2, 3], p=[0.1, 0.5, 0.3, 0.1])
                
                if num_tours_booked > 0 and tours:
                    # Try to get tours from the same destination if there's a hotel booking
                    possible_tours = []
                    
                    if 'destination_id' in locals():
                        # Get tours for this destination
                        for tour in tours:
                            if tour[1] == destination_id:
                                possible_tours.append(tour)
                    
                    # If not enough tours in that destination, use any tours
                    if len(possible_tours) < num_tours_booked:
                        possible_tours = tours
                    
                    # Sample some tours
                    selected_tours = random.sample(possible_tours, min(num_tours_booked, len(possible_tours)))
                    
                    for tour in selected_tours:
                        tour_id = tour[0]
                        tour_price = tour[2]
                        
                        # Tour date is during the hotel stay if there's a hotel booking
                        if 'check_in_date' in locals() and 'check_out_date' in locals():
                            check_in = datetime.strptime(check_in_date, '%Y-%m-%d')
                            check_out = datetime.strptime(check_out_date, '%Y-%m-%d')
                            days_range = (check_out - check_in).days
                            
                            if days_range > 0:
                                tour_day = random.randint(0, days_range)
                                tour_date = (check_in + timedelta(days=tour_day)).strftime('%Y-%m-%d')
                            else:
                                tour_date = check_in.strftime('%Y-%m-%d')
                        else:
                            # If no hotel booking, tour happens around the trip start date
                            days_offset = random.randint(-3, 3)
                            tour_date = (datetime.strptime(trip_start_date, '%Y-%m-%d') + timedelta(days=days_offset)).strftime('%Y-%m-%d')
                        
                        num_participants = random.randint(1, 6)
                        tour_cost = tour_price * num_participants
                        total_cost += tour_cost
                        
                        tour_booking_id = len(tour_bookings) + 1
                        
                        # Insert tour booking
                        cursor.execute("""
                            INSERT INTO tour_bookings 
                                (tour_booking_id, booking_id, tour_id, tour_date, number_of_participants, special_requirements)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, 
                            (tour_booking_id, booking_id, tour_id, tour_date, num_participants,
                            "None" if random.random() < 0.8 else "Dietary restrictions")
                        )
                        
                        tour_bookings.append(tour_booking_id)
                        
                        # Generate review for tour if booking is completed
                        if status == 'completed' and random.random() < 0.6:  # 60% chance of leaving a review
                            # Review happens after tour
                            days_after_tour = random.randint(1, 15)
                            review_date = (datetime.strptime(tour_date, '%Y-%m-%d') + timedelta(days=days_after_tour)).strftime('%Y-%m-%d')
                            
                            # Get tour name
                            cursor.execute("SELECT name FROM tours WHERE tour_id = ?", (tour_id,))
                            tour_name = cursor.fetchone()[0]
                            
                            # Rating tends to be high but can be low
                            if random.random() < 0.85:  # 85% happy customers for tours
                                rating = random.randint(4, 5)
                                comment = f"Excellent {tour_name} tour! Highly recommended."
                            else:
                                rating = random.randint(1, 3)
                                comment = f"Not impressed with the {tour_name} tour."
                            
                            review_id = len(reviews) + 1
                            
                            # Insert review
                            cursor.execute("""
                                INSERT INTO reviews 
                                    (review_id, user_id, entity_type, entity_id, rating, comment, review_date, helpful_votes)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, 
                                (review_id, user_id, 'tour', tour_id, rating, comment, review_date, random.randint(0, 30))
                            )
                            
                            reviews.append(review_id)
                
                # Update the total cost in the bookings table
                cursor.execute("UPDATE bookings SET total_cost = ? WHERE booking_id = ?", 
                            (round(total_cost, 2), booking_id))
            
            # Commit the batch transaction
            conn.commit()
            print(f"  Processed bookings {start_idx} to {end_idx}...")
            
        except Exception as e:
            # Rollback the transaction if there was an error
            conn.rollback()
            print(f"Error processing bookings batch {batch+1}: {e}")
    
    # Generate user interactions
    print("Generating user interactions...")
    interaction_types = ['search', 'view', 'save', 'rate']
    entity_types = ['hotel', 'tour', 'destination']
    
    num_interactions = 10000
    interactions_batch_size = 1000
    num_interaction_batches = (num_interactions + interactions_batch_size - 1) // interactions_batch_size
    
    for batch in range(num_interaction_batches):
        # Start a transaction for this interactions batch
        cursor.execute("BEGIN TRANSACTION")
        try:
            start_idx = batch * interactions_batch_size + 1
            end_idx = min((batch + 1) * interactions_batch_size, num_interactions)
            
            for i in range(start_idx, end_idx + 1):
                user_id = random.choice(user_ids)
                interaction_type = random.choice(interaction_types)
                entity_type = random.choice(entity_types)
                
                # Pick a random entity based on type
                if entity_type == 'hotel':
                    entity_id = random.choice(hotels)[0]
                elif entity_type == 'tour' and tours:
                    entity_id = random.choice(tours)[0]
                else:  # destination
                    entity_id = random.choice(destination_ids)
                
                # Generate random timestamp in last 90 days
                days_ago = random.randint(0, 90)
                hours_ago = random.randint(0, 23)
                minutes_ago = random.randint(0, 59)
                seconds_ago = random.randint(0, 59)
                
                timestamp = (datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago, seconds=seconds_ago)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Generate interaction details based on type
                if interaction_type == 'search':
                    # Get name of entity
                    if entity_type == 'hotel':
                        cursor.execute("SELECT name FROM hotels WHERE hotel_id = ?", (entity_id,))
                        entity_name = cursor.fetchone()[0]
                    elif entity_type == 'tour':
                        cursor.execute("SELECT name FROM tours WHERE tour_id = ?", (entity_id,))
                        entity_name = cursor.fetchone()[0]
                    else:  # destination
                        cursor.execute("SELECT name FROM destinations WHERE destination_id = ?", (entity_id,))
                        entity_name = cursor.fetchone()[0]
                        
                    details = json.dumps({"search_terms": f"{'hotels' if entity_type == 'hotel' else 'tours' if entity_type == 'tour' else 'destinations'} in {entity_name}"})
                elif interaction_type == 'view':
                    details = json.dumps({"view_duration_seconds": random.randint(5, 300)})
                elif interaction_type == 'save':
                    details = json.dumps({"saved_to_list": random.choice(["Favorites", "Wish List", "To Visit", "Summer Vacation"])})
                else:  # rate
                    details = json.dumps({"rating": random.randint(1, 5)})
                
                # Insert interaction
                cursor.execute("""
                    INSERT INTO user_interactions 
                        (interaction_id, user_id, interaction_type, entity_type, entity_id, timestamp, interaction_details, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, 
                    (i, user_id, interaction_type, entity_type, entity_id, timestamp, details, f"session_{i % 1000 + 1}")
                )
            
            # Commit this batch of interactions
            conn.commit()
            print(f"  Processed interactions {start_idx} to {end_idx}...")
            
        except Exception as e:
            # Rollback the transaction if there was an error
            conn.rollback()
            print(f"Error processing interactions batch {batch+1}: {e}")
    
    print(f"Generated {num_interactions} user interactions.")

def setup_pyodbc_connection(conn_str=None):
    """
    Test a PyODBC connection to SQL Server
    
    Args:
        conn_str: Connection string to test
        
    Returns:
        Boolean indicating success
    """
    if conn_str is None:
        # Default connection string
        conn_str = (
            "DRIVER=/opt/homebrew/lib/libtdsodbc.so;"
            "SERVER=127.0.0.1;"
            "PORT=1433;"
            "DATABASE=virtuoso_travel;"
            "UID=climbing_user;"
            "PWD=hoosierheights;"
            "TDS_Version=7.4;"
        )
    
    try:
        # Try to establish connection
        conn = pyodbc.connect(conn_str)
        print("Successfully connected to database using PyODBC")
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        print(f"SQL Server version: {version}")
        
        cursor.close()
        conn.close()
        return True
    except pyodbc.Error as e:
        print(f"PyODBC connection error: {e}")
        return False

if __name__ == "__main__":
    # Try to create and populate the database
    print("Setting up the Virtuoso Travel database...")
    
    # Ask if the user wants to use existing API data or generate new data
    conn_str = create_database("virtuoso_travel")
    
    # Test the connection
    success = setup_pyodbc_connection(conn_str)
    
    if not success:
        print("\nError: Unable to connect to the database.")
    else:
        print("\nDatabase setup complete and connection successful.") 