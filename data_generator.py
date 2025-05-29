import os
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
import json
import pyodbc
from faker import Faker

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create output directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Database connection function
def get_db_connection():
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

# Generate Users data
def generate_users(num_users=1000, output_file=None, db_conn=None):
    """
    Generate user data including personal information and preferences.
    
    Args:
        num_users (int): Number of users to generate
        output_file (str): Path to output CSV file (optional)
        db_conn: Database connection object (optional)
        
    Returns:
        DataFrame: Generated user data
    """
    try:
        # Languages
        languages = ['English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese', 'Arabic']
        
        # Countries for residence
        countries = ['United States', 'Canada', 'United Kingdom', 'France', 'Germany', 
                    'Spain', 'Italy', 'Japan', 'Australia', 'Brazil', 'Mexico', 
                    'China', 'India', 'Russia', 'South Africa']
        
        # Loyalty tiers
        loyalty_tiers = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
        
        # Determine starting user_id by checking the database
        start_user_id = 1
        if db_conn:
            try:
                cursor = db_conn.cursor()
                cursor.execute("SELECT MAX(user_id) FROM users")
                max_id = cursor.fetchone()[0]
                if max_id is not None:
                    start_user_id = max_id + 1
                    print(f"Starting user_id generation from {start_user_id}")
            except Exception as e:
                print(f"Error checking existing user IDs: {str(e)}")
                # Continue with default start_user_id = 1
        
        # Generate user data
        users = []
        for i in range(num_users):
            user_id = start_user_id + i
            
            # Personal information
            first_name = fake.first_name()
            last_name = fake.last_name()
            email = fake.email()
            date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=80)
            signup_date = fake.date_between(start_date='-5y', end_date='today')
            preferred_language = random.choice(languages)
            country_of_residence = random.choice(countries)
            loyalty_tier = random.choice(loyalty_tiers)
            
            user = {
                'user_id': user_id,
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'date_of_birth': date_of_birth,
                'signup_date': signup_date,
                'preferred_language': preferred_language,
                'country_of_residence': country_of_residence,
                'loyalty_tier': loyalty_tier
            }
            users.append(user)
        
        # Convert to DataFrame
        users_df = pd.DataFrame(users)
        
        # Save to file if output_file is provided
        if output_file:
            users_df.to_csv(output_file, index=False)
            print(f"Generated {num_users} users and saved to {output_file}")
        
        return users_df
    
    except Exception as e:
        print(f"Error generating users: {str(e)}")
        return pd.DataFrame()

# Generate Destinations data
def generate_destinations(num_destinations=200):
    # Popular regions and countries
    regions_countries = {
        'North America': ['USA', 'Canada', 'Mexico'],
        'Europe': ['France', 'Italy', 'Spain', 'UK', 'Germany', 'Greece', 'Portugal', 'Switzerland'],
        'Asia': ['Japan', 'Thailand', 'Vietnam', 'China', 'Singapore', 'South Korea', 'Indonesia', 'Malaysia'],
        'Oceania': ['Australia', 'New Zealand', 'Fiji'],
        'South America': ['Brazil', 'Argentina', 'Peru', 'Chile', 'Colombia'],
        'Africa': ['South Africa', 'Morocco', 'Egypt', 'Kenya', 'Tanzania'],
        'Caribbean': ['Jamaica', 'Bahamas', 'Dominican Republic', 'Cuba']
    }
    
    climate_types = ['Tropical', 'Mediterranean', 'Desert', 'Continental', 'Temperate', 'Arctic', 'Alpine']
    seasons = ['Spring', 'Summer', 'Fall', 'Winter', 'Year-round']
    
    destinations = []
    destination_id = 1
    
    for region, countries in regions_countries.items():
        for country in countries:
            # Generate multiple destinations per country
            num_places = random.randint(2, 10)
            for _ in range(num_places):
                # Generate random coordinates within realistic bounds
                if region == 'North America':
                    lat = random.uniform(25, 50)
                    lon = random.uniform(-125, -70)
                elif region == 'Europe':
                    lat = random.uniform(35, 60)
                    lon = random.uniform(-10, 30)
                elif region == 'Asia':
                    lat = random.uniform(0, 45)
                    lon = random.uniform(70, 145)
                elif region == 'Oceania':
                    lat = random.uniform(-45, -10)
                    lon = random.uniform(110, 180)
                elif region == 'South America':
                    lat = random.uniform(-40, 10)
                    lon = random.uniform(-80, -35)
                elif region == 'Africa':
                    lat = random.uniform(-30, 35)
                    lon = random.uniform(-20, 50)
                else:  # Caribbean
                    lat = random.uniform(10, 25)
                    lon = random.uniform(-85, -60)
                
                destinations.append({
                    'destination_id': destination_id,
                    'name': f"{country} Destination {destination_id}",
                    'country': country,
                    'region': region,
                    'latitude': round(lat, 6),
                    'longitude': round(lon, 6),
                    'popularity_score': round(random.uniform(1, 10), 1),
                    'average_rating': round(random.uniform(3, 5), 1),
                    'climate_type': random.choice(climate_types),
                    'best_season_to_visit': random.choice(seasons),
                    'description': f"Beautiful destination in {country} with amazing attractions."
                })
                
                destination_id += 1
                if destination_id > num_destinations:
                    break
            if destination_id > num_destinations:
                break
        if destination_id > num_destinations:
            break
    
    return pd.DataFrame(destinations[:num_destinations])

# Generate Hotels data
def generate_hotels(destinations_df, num_hotels=500):
    hotel_chains = ['Marriott', 'Hilton', 'Hyatt', 'InterContinental', 'Accor', 'Choice', 'Best Western', 'Wyndham']
    hotel_types = ['Resort', 'Hotel', 'Boutique', 'Lodge', 'Inn', 'Suites']
    
    hotels = []
    
    for hotel_id in range(1, num_hotels + 1):
        # Pick a random destination
        destination = destinations_df.sample(1).iloc[0]
        
        # Slight variation in hotel coordinates from destination
        lat_variation = random.uniform(-0.05, 0.05)
        lon_variation = random.uniform(-0.05, 0.05)
        
        chain = random.choice(hotel_chains)
        hotel_type = random.choice(hotel_types)
        
        hotels.append({
            'hotel_id': hotel_id,
            'name': f"{chain} {hotel_type} {destination['name']}",
            'destination_id': destination['destination_id'],
            'star_rating': random.randint(1, 5),
            'price_category': random.choice(['$', '$$', '$$$', '$$$$', '$$$$$']),
            'has_pool': random.choice([True, False]),
            'has_spa': random.choice([True, False]),
            'has_restaurant': random.choice([True, False]),
            'rooms_available': random.randint(10, 500),
            'average_rating': round(random.uniform(2.5, 5.0), 1),
            'address': f"{random.randint(1, 999)} Main Street",
            'latitude': round(destination['latitude'] + lat_variation, 6),
            'longitude': round(destination['longitude'] + lon_variation, 6)
        })
    
    return pd.DataFrame(hotels)

# Generate Tours data
def generate_tours(destinations_df, num_tours=800):
    tour_types = ['Adventure', 'Cultural', 'Culinary', 'Historical', 'Nature', 'Relaxation', 'Religious', 'Wildlife']
    difficulty_levels = ['Easy', 'Moderate', 'Challenging', 'Difficult', 'Extreme']
    
    tours = []
    
    for tour_id in range(1, num_tours + 1):
        # Pick a random destination
        destination = destinations_df.sample(1).iloc[0]
        tour_type = random.choice(tour_types)
        
        tours.append({
            'tour_id': tour_id,
            'name': f"{destination['name']} {tour_type} Experience",
            'destination_id': destination['destination_id'],
            'duration_hours': random.randint(1, 72),
            'price': round(random.uniform(20, 2000), 2),
            'category': tour_type,
            'group_size_limit': random.randint(4, 50),
            'difficulty_level': random.choice(difficulty_levels),
            'average_rating': round(random.uniform(3.0, 5.0), 1),
            'description': f"Enjoy this amazing {tour_type.lower()} tour in {destination['name']}!",
            'availability_schedule': random.choice(['Daily', 'Weekdays', 'Weekends', 'Seasonal'])
        })
    
    return pd.DataFrame(tours)

# Generate User Preferences
def generate_user_preferences(users_df):
    preferences = []
    
    for _, user in users_df.iterrows():
        preferred_activities = random.sample(['Adventure', 'Cultural', 'Culinary', 'Historical', 'Nature', 'Relaxation', 'Religious', 'Wildlife'], 
                                           random.randint(1, 5))
        
        preferred_climates = random.sample(['Tropical', 'Mediterranean', 'Desert', 'Continental', 'Temperate', 'Arctic', 'Alpine'], 
                                          random.randint(1, 3))
        
        preferences.append({
            'preference_id': user['user_id'],
            'user_id': user['user_id'],
            'preferred_hotel_stars': random.randint(1, 5),
            'preferred_budget_category': random.choice(['Budget', 'Moderate', 'Luxury']),
            'preferred_activities': json.dumps(preferred_activities),
            'preferred_climates': json.dumps(preferred_climates),
            'travel_style': random.choice(['Solo', 'Couple', 'Family', 'Group']),
            'maximum_flight_duration': random.choice([4, 8, 12, 24, 48])
        })
    
    return pd.DataFrame(preferences)

# Generate bookings, hotel bookings, tour bookings, and reviews
def generate_transaction_data(users_df, hotels_df, tours_df, num_bookings=3000):
    # Create bookings
    bookings = []
    hotel_bookings = []
    tour_bookings = []
    reviews = []
    
    booking_channels = ['web', 'mobile', 'agent']
    booking_statuses = ['confirmed', 'canceled', 'completed']
    room_types = ['Standard', 'Deluxe', 'Suite', 'Family', 'Executive']
    review_id = 1
    
    for booking_id in range(1, num_bookings + 1):
        # Pick a random user
        user = users_df.sample(1).iloc[0]
        
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
        
        bookings.append({
            'booking_id': booking_id,
            'user_id': user['user_id'],
            'booking_date': booking_date,
            'total_cost': total_cost,  # Placeholder, will update later
            'payment_status': 'paid' if status != 'canceled' else 'refunded',
            'booking_status': status,
            'booking_channel': random.choice(booking_channels)
        })
        
        # Randomly decide if this booking includes a hotel
        if random.random() < 0.9:  # 90% of bookings include hotel
            hotel = hotels_df.sample(1).iloc[0]
            stay_length = random.randint(1, 14)  # 1-14 nights
            
            check_in_date = trip_start_date
            check_out_date = (datetime.strptime(check_in_date, '%Y-%m-%d') + timedelta(days=stay_length)).strftime('%Y-%m-%d')
            
            rate_per_night = float(hotel['price_category'].count('$')) * 100 + random.uniform(0, 100)
            hotel_cost = rate_per_night * stay_length
            total_cost += hotel_cost
            
            hotel_bookings.append({
                'hotel_booking_id': len(hotel_bookings) + 1,
                'booking_id': booking_id,
                'hotel_id': hotel['hotel_id'],
                'check_in_date': check_in_date,
                'check_out_date': check_out_date,
                'room_type': random.choice(room_types),
                'number_of_guests': random.randint(1, 6),
                'special_requests': "None" if random.random() < 0.7 else "Late check-out requested",
                'rate_per_night': round(rate_per_night, 2)
            })
            
            # Generate review for hotel if booking is completed
            if status == 'completed' and random.random() < 0.7:  # 70% chance of leaving a review
                # Review happens after check-out
                days_after_checkout = random.randint(1, 30)
                review_date = (datetime.strptime(check_out_date, '%Y-%m-%d') + timedelta(days=days_after_checkout)).strftime('%Y-%m-%d')
                
                # Rating tends to be high but can be low
                if random.random() < 0.8:  # 80% happy customers
                    rating = random.randint(4, 5)
                    comment = f"Great stay at {hotel['name']}! Would recommend."
                else:
                    rating = random.randint(1, 3)
                    comment = f"Disappointing experience at {hotel['name']}."
                
                reviews.append({
                    'review_id': review_id,
                    'user_id': user['user_id'],
                    'entity_type': 'hotel',
                    'entity_id': hotel['hotel_id'],
                    'rating': rating,
                    'comment': comment,
                    'review_date': review_date,
                    'helpful_votes': random.randint(0, 50)
                })
                review_id += 1
        
        # Randomly decide if this booking includes tours
        num_tours_booked = np.random.choice([0, 1, 2, 3], p=[0.1, 0.5, 0.3, 0.1])
        
        if num_tours_booked > 0:
            # Get tours from the same destination if possible
            if len(hotel_bookings) > 0:
                hotel_id = hotel_bookings[-1]['hotel_id']
                hotel_row = hotels_df[hotels_df['hotel_id'] == hotel_id].iloc[0]
                destination_id = hotel_row['destination_id']
                possible_tours = tours_df[tours_df['destination_id'] == destination_id]
                
                # If not enough tours in that destination, select random tours
                if len(possible_tours) < num_tours_booked:
                    possible_tours = tours_df
            else:
                possible_tours = tours_df
            
            tours_for_booking = possible_tours.sample(min(num_tours_booked, len(possible_tours)))
            
            for _, tour in tours_for_booking.iterrows():
                # Tour date is during the hotel stay if there's a hotel booking
                if len(hotel_bookings) > 0:
                    check_in = datetime.strptime(hotel_bookings[-1]['check_in_date'], '%Y-%m-%d')
                    check_out = datetime.strptime(hotel_bookings[-1]['check_out_date'], '%Y-%m-%d')
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
                tour_cost = tour['price'] * num_participants
                total_cost += tour_cost
                
                tour_bookings.append({
                    'tour_booking_id': len(tour_bookings) + 1,
                    'booking_id': booking_id,
                    'tour_id': tour['tour_id'],
                    'tour_date': tour_date,
                    'number_of_participants': num_participants,
                    'special_requirements': "None" if random.random() < 0.8 else "Dietary restrictions"
                })
                
                # Generate review for tour if booking is completed
                if status == 'completed' and random.random() < 0.6:  # 60% chance of leaving a review
                    # Review happens after tour
                    days_after_tour = random.randint(1, 15)
                    review_date = (datetime.strptime(tour_date, '%Y-%m-%d') + timedelta(days=days_after_tour)).strftime('%Y-%m-%d')
                    
                    # Rating tends to be high but can be low
                    if random.random() < 0.85:  # 85% happy customers for tours
                        rating = random.randint(4, 5)
                        comment = f"Excellent {tour['name']} tour! Highly recommended."
                    else:
                        rating = random.randint(1, 3)
                        comment = f"Not impressed with the {tour['name']} tour."
                    
                    reviews.append({
                        'review_id': review_id,
                        'user_id': user['user_id'],
                        'entity_type': 'tour',
                        'entity_id': tour['tour_id'],
                        'rating': rating,
                        'comment': comment,
                        'review_date': review_date,
                        'helpful_votes': random.randint(0, 30)
                    })
                    review_id += 1
        
        # Update the total cost in the bookings table
        bookings[-1]['total_cost'] = round(total_cost, 2)
    
    return (pd.DataFrame(bookings), 
            pd.DataFrame(hotel_bookings), 
            pd.DataFrame(tour_bookings), 
            pd.DataFrame(reviews))

# Generate User Interactions data
def generate_user_interactions(users_df, hotels_df, tours_df, destinations_df, num_interactions=10000):
    interaction_types = ['search', 'view', 'save', 'rate']
    entity_types = ['hotel', 'tour', 'destination']
    
    interactions = []
    
    for i in range(1, num_interactions + 1):
        user = users_df.sample(1).iloc[0]
        interaction_type = random.choice(interaction_types)
        entity_type = random.choice(entity_types)
        
        # Pick a random entity based on type
        if entity_type == 'hotel':
            entity = hotels_df.sample(1).iloc[0]
            entity_id = entity['hotel_id']
        elif entity_type == 'tour':
            entity = tours_df.sample(1).iloc[0]
            entity_id = entity['tour_id']
        else:  # destination
            entity = destinations_df.sample(1).iloc[0]
            entity_id = entity['destination_id']
        
        # Generate random timestamp in last 90 days
        days_ago = random.randint(0, 90)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        seconds_ago = random.randint(0, 59)
        
        timestamp = (datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago, seconds=seconds_ago)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate interaction details based on type
        if interaction_type == 'search':
            details = json.dumps({"search_terms": f"{'hotels' if entity_type == 'hotel' else 'tours' if entity_type == 'tour' else 'destinations'} in {entity['name'] if entity_type == 'destination' else (entity['name'] if 'name' in entity else '')}"})
        elif interaction_type == 'view':
            details = json.dumps({"view_duration_seconds": random.randint(5, 300)})
        elif interaction_type == 'save':
            details = json.dumps({"saved_to_list": random.choice(["Favorites", "Wish List", "To Visit", "Summer Vacation"])})
        else:  # rate
            details = json.dumps({"rating": random.randint(1, 5)})
        
        interactions.append({
            'interaction_id': i,
            'user_id': user['user_id'],
            'interaction_type': interaction_type,
            'entity_type': entity_type,
            'entity_id': entity_id,
            'timestamp': timestamp,
            'interaction_details': details,
            'session_id': f"session_{i % 1000 + 1}"
        })
    
    return pd.DataFrame(interactions)

# Main function to generate all data and save to CSV
def generate_all_data():
    print("Generating users data...")
    users = generate_users(1000)
    users.to_csv('data/users.csv', index=False)
    
    print("Generating destinations data...")
    destinations = generate_destinations(200)
    destinations.to_csv('data/destinations.csv', index=False)
    
    print("Generating hotels data...")
    hotels = generate_hotels(destinations, 500)
    hotels.to_csv('data/hotels.csv', index=False)
    
    print("Generating tours data...")
    tours = generate_tours(destinations, 800)
    tours.to_csv('data/tours.csv', index=False)
    
    print("Generating user preferences...")
    preferences = generate_user_preferences(users)
    preferences.to_csv('data/user_preferences.csv', index=False)
    
    print("Generating bookings and reviews...")
    bookings, hotel_bookings, tour_bookings, reviews = generate_transaction_data(users, hotels, tours, 3000)
    bookings.to_csv('data/bookings.csv', index=False)
    hotel_bookings.to_csv('data/hotel_bookings.csv', index=False)
    tour_bookings.to_csv('data/tour_bookings.csv', index=False)
    reviews.to_csv('data/reviews.csv', index=False)
    
    print("Generating user interactions...")
    interactions = generate_user_interactions(users, hotels, tours, destinations, 10000)
    interactions.to_csv('data/user_interactions.csv', index=False)
    
    print("Data generation complete!")
    
    return {
        "users": users,
        "destinations": destinations,
        "hotels": hotels,
        "tours": tours,
        "preferences": preferences,
        "bookings": bookings,
        "hotel_bookings": hotel_bookings,
        "tour_bookings": tour_bookings,
        "reviews": reviews,
        "interactions": interactions
    }

# Function to generate data and load it into the database
def generate_data():
    """Generate simulation data and load it into the database."""
    try:
        print("Generating simulation data...")
        data = generate_all_data()
        
        print("Loading data into database...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert Users
        print("Inserting users...")
        for _, user in data["users"].iterrows():
            cursor.execute("""
                INSERT INTO users 
                (user_id, first_name, last_name, email, date_of_birth, signup_date, 
                preferred_language, country_of_residence, loyalty_tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            user['user_id'], user['first_name'], user['last_name'], user['email'], 
            user['date_of_birth'], user['signup_date'], user['preferred_language'], 
            user['country_of_residence'], user['loyalty_tier'])
        
        # Insert Destinations
        print("Inserting destinations...")
        for _, dest in data["destinations"].iterrows():
            cursor.execute("""
                INSERT INTO destinations 
                (destination_id, name, country, region, latitude, longitude, 
                popularity_score, average_rating, climate_type, best_season_to_visit, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            dest['destination_id'], dest['name'], dest['country'], dest['region'], 
            dest['latitude'], dest['longitude'], dest['popularity_score'], 
            dest['average_rating'], dest['climate_type'], dest['best_season_to_visit'], 
            dest['description'])
        
        # Insert Hotels
        print("Inserting hotels...")
        for _, hotel in data["hotels"].iterrows():
            cursor.execute("""
                INSERT INTO hotels 
                (hotel_id, name, destination_id, star_rating, price_category, 
                has_pool, has_spa, has_restaurant, rooms_available, average_rating, 
                address, latitude, longitude)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            hotel['hotel_id'], hotel['name'], hotel['destination_id'], hotel['star_rating'], 
            hotel['price_category'], hotel['has_pool'], hotel['has_spa'], hotel['has_restaurant'], 
            hotel['rooms_available'], hotel['average_rating'], hotel['address'], 
            hotel['latitude'], hotel['longitude'])
        
        # Insert Tours
        print("Inserting tours...")
        for _, tour in data["tours"].iterrows():
            cursor.execute("""
                INSERT INTO tours 
                (tour_id, name, destination_id, duration_hours, price, category, 
                group_size_limit, difficulty_level, average_rating, description, availability_schedule)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            tour['tour_id'], tour['name'], tour['destination_id'], tour['duration_hours'], 
            tour['price'], tour['category'], tour['group_size_limit'], tour['difficulty_level'], 
            tour['average_rating'], tour['description'], tour['availability_schedule'])
        
        # Insert User Preferences
        print("Inserting user preferences...")
        for _, pref in data["preferences"].iterrows():
            cursor.execute("""
                INSERT INTO user_preferences 
                (preference_id, user_id, preferred_hotel_stars, preferred_budget_category, 
                preferred_activities, preferred_climates, travel_style, maximum_flight_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            pref['preference_id'], pref['user_id'], pref['preferred_hotel_stars'], 
            pref['preferred_budget_category'], pref['preferred_activities'], 
            pref['preferred_climates'], pref['travel_style'], pref['maximum_flight_duration'])
        
        # Insert Bookings
        print("Inserting bookings...")
        for _, booking in data["bookings"].iterrows():
            cursor.execute("""
                INSERT INTO bookings 
                (booking_id, user_id, booking_date, total_cost, payment_status, booking_status, booking_channel)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, 
            booking['booking_id'], booking['user_id'], booking['booking_date'], 
            booking['total_cost'], booking['payment_status'], booking['booking_status'], 
            booking['booking_channel'])
        
        # Insert Hotel Bookings
        print("Inserting hotel bookings...")
        for _, hotel_booking in data["hotel_bookings"].iterrows():
            cursor.execute("""
                INSERT INTO hotel_bookings 
                (hotel_booking_id, booking_id, hotel_id, check_in_date, check_out_date, 
                room_type, number_of_guests, special_requests, rate_per_night)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            hotel_booking['hotel_booking_id'], hotel_booking['booking_id'], hotel_booking['hotel_id'], 
            hotel_booking['check_in_date'], hotel_booking['check_out_date'], hotel_booking['room_type'], 
            hotel_booking['number_of_guests'], hotel_booking['special_requests'], 
            hotel_booking['rate_per_night'])
        
        # Insert Tour Bookings
        print("Inserting tour bookings...")
        for _, tour_booking in data["tour_bookings"].iterrows():
            cursor.execute("""
                INSERT INTO tour_bookings 
                (tour_booking_id, booking_id, tour_id, tour_date, 
                number_of_participants, special_requirements)
                VALUES (?, ?, ?, ?, ?, ?)
            """, 
            tour_booking['tour_booking_id'], tour_booking['booking_id'], tour_booking['tour_id'], 
            tour_booking['tour_date'], tour_booking['number_of_participants'], 
            tour_booking['special_requirements'])
        
        # Insert Reviews
        print("Inserting reviews...")
        for _, review in data["reviews"].iterrows():
            cursor.execute("""
                INSERT INTO reviews 
                (review_id, user_id, entity_type, entity_id, rating, comment, review_date, helpful_votes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            review['review_id'], review['user_id'], review['entity_type'], review['entity_id'], 
            review['rating'], review['comment'], review['review_date'], review['helpful_votes'])
        
        # Insert User Interactions
        print("Inserting user interactions...")
        for _, interaction in data["interactions"].iterrows():
            cursor.execute("""
                INSERT INTO user_interactions 
                (interaction_id, user_id, interaction_type, entity_type, 
                entity_id, timestamp, interaction_details, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            interaction['interaction_id'], interaction['user_id'], interaction['interaction_type'], 
            interaction['entity_type'], interaction['entity_id'], interaction['timestamp'], 
            interaction['interaction_details'], interaction['session_id'])
        
        conn.commit()
        print("All data inserted successfully!")
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error generating or inserting data: {e}")
        return False

if __name__ == "__main__":
    generate_all_data() 