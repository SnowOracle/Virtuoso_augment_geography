import os
import requests
import pandas as pd
import json
import time
from datetime import datetime
import random
import pyodbc

# Create a directory for API data
if not os.path.exists('api_data'):
    os.makedirs('api_data')

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

def clear_existing_data():
    """
    Clear existing data from all tables in the database before inserting new data.
    This ensures we don't have duplicates or inconsistencies.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("Clearing existing data from database...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Disable foreign key constraints temporarily
        cursor.execute("EXEC sp_MSforeachtable 'ALTER TABLE ? NOCHECK CONSTRAINT ALL'")
        
        # Clear data from tables in reverse dependency order
        tables = [
            "user_interactions",
            "reviews",
            "tour_bookings",
            "hotel_bookings",
            "bookings",
            "user_preferences",
            "tours",
            "hotels",
            "destinations",
            "users"
        ]
        
        for table in tables:
            print(f"Clearing data from {table}...")
            cursor.execute(f"DELETE FROM {table}")
        
        # Re-enable foreign key constraints
        cursor.execute("EXEC sp_MSforeachtable 'ALTER TABLE ? CHECK CONSTRAINT ALL'")
        
        conn.commit()
        conn.close()
        print("All existing data cleared successfully.")
        return True
    except Exception as e:
        print(f"Error clearing existing data: {e}")
        return False

def fetch_opentrip_map_data(limit=1000, api_key=None):
    """
    Fetch destination data from OpenTripMap API with optimized performance
    
    Args:
        limit: Maximum number of destinations to retrieve
        api_key: OpenTripMap API key, defaults to environment variable
    
    Returns:
        DataFrame with destination data
    """
    import concurrent.futures
    
    if api_key is None:
        # Try to get API key from environment variable
        api_key = os.environ.get('OPENTRIP_API_KEY')
        if api_key is None:
            print("Warning: No OpenTripMap API key provided. Using limited free tier.")
            api_key = "5ae2e3f221c38a28845f05b67f0a2f46448d26f2f8f2d1e9af537c25"  # Default demo key with limits
    
    base_url = "https://api.opentripmap.com/0.1/en/places"
    
    # First, get a list of countries to ensure global coverage
    print("Fetching countries list...")
    # Create a session object to reuse HTTP connections
    session = requests.Session()
    
    countries_url = f"{base_url}/countries?apikey={api_key}"
    response = session.get(countries_url)
    countries = response.json()
    
    # Prepare to collect destinations
    all_destinations = []
    destination_id = 1
    
    # Define kinds of places to look for
    place_kinds = ["interesting_places", "tourist_facilities", "natural", "cultural", 
                  "architecture", "historic", "museums", "religion", "beaches"]
    
    # Process countries in random order to get better global distribution
    random.shuffle(countries)
    
    print(f"Fetching destinations from {len(countries)} countries...")
    
    # Process countries in parallel with a ThreadPoolExecutor
    # This can significantly speed up the data collection
    def process_country(country):
        country_destinations = []
        local_session = requests.Session()  # Create a session for each thread
        
        country_code = country["country"]
        country_name = country["name"]
        
        print(f"Fetching destinations in {country_name}...")
        
        # For each country, try to get some top destinations
        for kind in random.sample(place_kinds, min(3, len(place_kinds))):
            # Get places of this kind in this country
            places_url = f"{base_url}/radius?radius=10000&limit=20&kind={kind}&format=json&country={country_code}&apikey={api_key}"
            try:
                response = local_session.get(places_url)
                places = response.json()
                
                # If no places found, continue to next kind
                if not places or isinstance(places, dict) and 'error' in places:
                    continue
                
                # Process each place
                for place in places:
                    # Get details for this place
                    xid = place.get("xid")
                    if not xid:
                        continue
                        
                    details_url = f"{base_url}/xid/{xid}?apikey={api_key}"
                    
                    try:
                        details_response = local_session.get(details_url)
                        details = details_response.json()
                        
                        # Skip if no detailed info or error
                        if 'error' in details or not details.get('name'):
                            continue
                        
                        # Extract relevant information
                        destination = {
                            'destination_id': None,  # Will be assigned globally later
                            'name': details.get('name', f"Destination"),
                            'country': country_name,
                            'region': details.get('address', {}).get('state', 'Unknown'),
                            'city': details.get('address', {}).get('city', ''),
                            'latitude': place.get('point', {}).get('lat', 0),
                            'longitude': place.get('point', {}).get('lon', 0),
                            'popularity_score': min(10, max(1, place.get('rate', 5) * 2)),  # Convert rate to 1-10 scale
                            'average_rating': min(5, max(3, place.get('rate', 4))),  # Convert to 3-5 scale
                            'climate_type': assign_climate_type(place.get('point', {}).get('lat', 0)),
                            'best_season_to_visit': assign_best_season(place.get('point', {}).get('lat', 0)),
                            'description': details.get('wikipedia_extracts', {}).get('text', 
                                         details.get('info', {}).get('descr', 
                                                               f"Beautiful destination in {country_name}.")),
                            'place_type': details.get('kinds', '').split(',')[0] if details.get('kinds') else kind,
                            'image_url': details.get('preview', {}).get('source', '')
                        }
                        
                        country_destinations.append(destination)
                            
                    except Exception as e:
                        print(f"Error fetching details for {xid}: {e}")
                        
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.2)
                    
            except Exception as e:
                print(f"Error fetching places in {country_name} for kind {kind}: {e}")
                
        # Add a small delay between countries to avoid overloading the API
        time.sleep(0.5)
        
        return country_destinations
    
    # Process countries in parallel with a limit to avoid overwhelming the API
    max_workers = min(10, len(countries))  # Use up to 10 workers but no more than countries count
    collected_destinations = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the process_country function to each country
        future_to_country = {executor.submit(process_country, country): country for country in countries[:min(30, len(countries))]}
        
        for future in concurrent.futures.as_completed(future_to_country):
            country = future_to_country[future]
            try:
                country_destinations = future.result()
                collected_destinations.extend(country_destinations)
                
                # Check if we've reached the limit
                if len(collected_destinations) >= limit:
                    break
            except Exception as e:
                print(f"Error processing country {country['name']}: {e}")
    
    # Assign unique IDs to the collected destinations
    for i, dest in enumerate(collected_destinations[:limit]):
        dest['destination_id'] = i + 1
    
    # Convert to DataFrame
    destinations_df = pd.DataFrame(collected_destinations[:limit])
    
    # Save the raw API data
    destinations_df.to_csv('api_data/destinations_api.csv', index=False)
    
    print(f"Fetched and saved {len(destinations_df)} destinations.")
    
    return destinations_df

def fetch_hotel_data(destinations_df, limit_per_destination=3, api_key=None):
    """
    Fetch hotel data from a public API for destinations
    
    Args:
        destinations_df: DataFrame with destination data
        limit_per_destination: Maximum number of hotels per destination
        api_key: API key
    
    Returns:
        DataFrame with hotel data
    """
    if api_key is None:
        # Try to get API key from environment variable
        api_key = os.environ.get('OPENTRIP_API_KEY')
        if api_key is None:
            print("Warning: No OpenTripMap API key provided. Using limited free tier.")
            api_key = "5ae2e3f221c38a28845f05b67f0a2f46448d26f2f8f2d1e9af537c25"  # Default demo key with limits
    
    base_url = "https://api.opentripmap.com/0.1/en/places"
    
    # Prepare to collect hotels
    all_hotels = []
    hotel_id = 1
    
    print(f"Fetching hotel data for {len(destinations_df)} destinations...")
    
    # Sample destinations if there are many
    if len(destinations_df) > 300:
        destinations_sample = destinations_df.sample(300)
    else:
        destinations_sample = destinations_df
        
    for _, destination in destinations_sample.iterrows():
        destination_id = destination['destination_id']
        lat = destination['latitude']
        lon = destination['longitude']
        
        print(f"Fetching hotels near {destination['name']}...")
        
        # Get accommodations near this destination
        hotels_url = f"{base_url}/radius?radius=10000&limit={limit_per_destination*2}&kind=accomodations&format=json&lat={lat}&lon={lon}&apikey={api_key}"
        
        try:
            response = requests.get(hotels_url)
            hotels = response.json()
            
            # If no hotels found, continue to next destination
            if not hotels or isinstance(hotels, dict) and 'error' in hotels:
                print(f"  No hotels found for {destination['name']}")
                continue
            
            # Process each hotel
            hotel_count = 0
            for hotel in hotels:
                # Get details for this hotel
                xid = hotel.get("xid")
                if not xid:
                    continue
                    
                details_url = f"{base_url}/xid/{xid}?apikey={api_key}"
                
                try:
                    details_response = requests.get(details_url)
                    details = details_response.json()
                    
                    # Skip if no detailed info or error
                    if 'error' in details or not details.get('name'):
                        continue
                    
                    # Extract relevant information
                    hotel_kinds = details.get('kinds', '').split(',')
                    
                    # Only process if it's actually an accommodation
                    if not any(k in ['accomodations', 'hotels', 'other_hotels', 'apartments', 'hostels', 'resorts'] 
                              for k in hotel_kinds):
                        continue
                    
                    # Generate star rating and amenities
                    star_rating = random.randint(2, 5)
                    has_pool = random.choice([True, False])
                    has_spa = random.choice([True, False]) if star_rating >= 3 else False
                    has_restaurant = True if star_rating >= 4 else random.choice([True, False])
                    
                    # Generate price category based on star rating
                    price_categories = ['$', '$$', '$$$', '$$$$', '$$$$$']
                    price_weights = [0.05, 0.1, 0.3, 0.4, 0.15]  # Higher weights for medium/high prices
                    
                    if star_rating <= 2:
                        price_weights = [0.6, 0.3, 0.1, 0, 0]
                    elif star_rating == 3:
                        price_weights = [0.1, 0.5, 0.3, 0.1, 0]
                    elif star_rating == 4:
                        price_weights = [0, 0.1, 0.4, 0.4, 0.1]
                    else:  # 5 stars
                        price_weights = [0, 0, 0.1, 0.4, 0.5]
                        
                    price_category = random.choices(price_categories, weights=price_weights)[0]
                    
                    # Extract hotel name, use fallbacks if needed
                    hotel_name = details.get('name', '')
                    if not hotel_name or hotel_name.strip() == '':
                        hotel_chains = ['Marriott', 'Hilton', 'Hyatt', 'InterContinental', 'Accor', 
                                      'Choice', 'Best Western', 'Wyndham']
                        hotel_types = ['Resort', 'Hotel', 'Boutique', 'Lodge', 'Inn', 'Suites']
                        
                        hotel_name = f"{random.choice(hotel_chains)} {random.choice(hotel_types)} {destination['name']}"
                    
                    # Create hotel entry
                    hotel_entry = {
                        'hotel_id': hotel_id,
                        'name': hotel_name,
                        'destination_id': destination_id,
                        'star_rating': star_rating,
                        'price_category': price_category,
                        'has_pool': has_pool,
                        'has_spa': has_spa,
                        'has_restaurant': has_restaurant,
                        'rooms_available': random.randint(20, 200),
                        'average_rating': min(5, max(2.5, details.get('rate', 4.0))),
                        'address': details.get('address', {}).get('road', f"{random.randint(1, 999)} Main Street"),
                        'latitude': hotel.get('point', {}).get('lat', 0),
                        'longitude': hotel.get('point', {}).get('lon', 0),
                        'image_url': details.get('preview', {}).get('source', '')
                    }
                    
                    all_hotels.append(hotel_entry)
                    hotel_id += 1
                    hotel_count += 1
                    
                    # If we've reached the limit for this destination, stop
                    if hotel_count >= limit_per_destination:
                        break
                        
                except Exception as e:
                    print(f"  Error fetching details for hotel {xid}: {e}")
                    
                # Add a small delay to avoid rate limiting
                time.sleep(0.2)
                
        except Exception as e:
            print(f"  Error fetching hotels for {destination['name']}: {e}")
        
        # Add a delay between destinations to avoid rate limiting
        time.sleep(1)
    
    # If we don't have enough hotels, generate some synthetic ones
    if len(all_hotels) < 500:
        print(f"Only found {len(all_hotels)} hotels, generating additional synthetic hotels...")
        
        hotel_chains = ['Marriott', 'Hilton', 'Hyatt', 'InterContinental', 'Accor', 
                      'Choice', 'Best Western', 'Wyndham', 'Four Seasons', 'Radisson']
        hotel_types = ['Resort', 'Hotel', 'Boutique', 'Lodge', 'Inn', 'Suites', 
                     'Grand', 'Luxury Collection', 'Spa Resort', 'Palace']
        
        needed_hotels = 500 - len(all_hotels)
        
        for _ in range(needed_hotels):
            # Select a random destination
            destination = destinations_df.sample(1).iloc[0]
            
            # Slight variation in hotel coordinates from destination
            lat_variation = random.uniform(-0.05, 0.05)
            lon_variation = random.uniform(-0.05, 0.05)
            
            chain = random.choice(hotel_chains)
            hotel_type = random.choice(hotel_types)
            star_rating = random.randint(1, 5)
            
            # Generate price category based on star rating
            price_categories = ['$', '$$', '$$$', '$$$$', '$$$$$']
            price_weights = [0.05, 0.1, 0.3, 0.4, 0.15]  # Higher weights for medium/high prices
            
            if star_rating <= 2:
                price_weights = [0.6, 0.3, 0.1, 0, 0]
            elif star_rating == 3:
                price_weights = [0.1, 0.5, 0.3, 0.1, 0]
            elif star_rating == 4:
                price_weights = [0, 0.1, 0.4, 0.4, 0.1]
            else:  # 5 stars
                price_weights = [0, 0, 0.1, 0.4, 0.5]
                
            price_category = random.choices(price_categories, weights=price_weights)[0]
            
            all_hotels.append({
                'hotel_id': hotel_id,
                'name': f"{chain} {hotel_type} {destination['name']}",
                'destination_id': destination['destination_id'],
                'star_rating': star_rating,
                'price_category': price_category,
                'has_pool': random.choice([True, False]),
                'has_spa': random.choice([True, False]) if star_rating >= 3 else False,
                'has_restaurant': True if star_rating >= 4 else random.choice([True, False]),
                'rooms_available': random.randint(10, 500),
                'average_rating': round(random.uniform(2.5, 5.0), 1),
                'address': f"{random.randint(1, 999)} Main Street",
                'latitude': round(destination['latitude'] + lat_variation, 6),
                'longitude': round(destination['longitude'] + lon_variation, 6),
                'image_url': ''
            })
            hotel_id += 1
    
    # Convert to DataFrame
    hotels_df = pd.DataFrame(all_hotels)
    
    # Save the data
    hotels_df.to_csv('api_data/hotels_api.csv', index=False)
    
    print(f"Fetched and generated a total of {len(hotels_df)} hotels.")
    
    return hotels_df

def assign_climate_type(latitude):
    """
    Assign a climate type based on latitude
    """
    climate_types = ['Tropical', 'Mediterranean', 'Desert', 'Continental', 'Temperate', 'Arctic', 'Alpine']
    
    latitude = abs(latitude)  # Convert to absolute value for simplicity
    
    if latitude < 15:
        return 'Tropical'
    elif 15 <= latitude < 25:
        return random.choice(['Tropical', 'Desert', 'Mediterranean'])
    elif 25 <= latitude < 35:
        return random.choice(['Mediterranean', 'Desert', 'Temperate'])
    elif 35 <= latitude < 50:
        return random.choice(['Temperate', 'Continental', 'Mediterranean'])
    elif 50 <= latitude < 65:
        return random.choice(['Continental', 'Temperate'])
    elif latitude >= 65:
        return random.choice(['Arctic', 'Continental'])
    
    # Randomly assign Alpine climate to some places regardless of latitude
    if random.random() < 0.1:
        return 'Alpine'
    
    return random.choice(climate_types)

def assign_best_season(latitude):
    """
    Assign a best season to visit based on latitude
    """
    seasons = ['Spring', 'Summer', 'Fall', 'Winter', 'Year-round']
    latitude = abs(latitude)  # Convert to absolute value for simplicity
    
    # Tropical regions are often good year-round or in winter (dry season)
    if latitude < 15:
        return random.choice(['Winter', 'Year-round'])
    
    # Southern regions often best in spring/fall to avoid extreme heat
    elif 15 <= latitude < 35:
        return random.choice(['Spring', 'Fall', 'Winter'])
    
    # Temperate regions often best in summer/fall
    elif 35 <= latitude < 60:
        return random.choice(['Spring', 'Summer', 'Fall'])
    
    # Northern regions often best in summer
    else:
        return 'Summer'

def generate_synthetic_user_data(num_users=10000):
    """
    Generate synthetic user data
    
    Args:
        num_users: Number of user profiles to generate
        
    Returns:
        DataFrame with user data
    """
    print(f"Generating {num_users} synthetic user profiles...")
    
    first_names = ['John', 'Emma', 'Michael', 'Olivia', 'William', 'Sophia', 'James', 'Ava', 'Alexander', 'Isabella',
                 'Benjamin', 'Mia', 'Elijah', 'Charlotte', 'Lucas', 'Amelia', 'Mason', 'Harper', 'Logan', 'Evelyn',
                 'Ethan', 'Abigail', 'Jacob', 'Emily', 'Jack', 'Elizabeth', 'Noah', 'Sofia', 'Daniel', 'Avery',
                 'Samuel', 'Ella', 'David', 'Scarlett', 'Joseph', 'Grace', 'Carter', 'Lily', 'Owen', 'Chloe',
                 'Jayden', 'Victoria', 'Gabriel', 'Madison', 'Matthew', 'Eleanor', 'Leo', 'Hannah', 'Lincoln', 'Lillian']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Garcia', 'Rodriguez', 'Wilson',
                'Martinez', 'Anderson', 'Taylor', 'Thomas', 'Hernandez', 'Moore', 'Martin', 'Jackson', 'Thompson', 'White',
                'Lopez', 'Lee', 'Gonzalez', 'Harris', 'Clark', 'Lewis', 'Robinson', 'Walker', 'Perez', 'Hall',
                'Young', 'Allen', 'Sanchez', 'Wright', 'King', 'Scott', 'Green', 'Baker', 'Adams', 'Nelson',
                'Hill', 'Ramirez', 'Campbell', 'Mitchell', 'Roberts', 'Carter', 'Phillips', 'Evans', 'Turner', 'Torres']
    
    languages = ['English', 'Spanish', 'French', 'German', 'Mandarin', 'Japanese', 'Italian', 'Portuguese', 
               'Hindi', 'Arabic', 'Russian', 'Korean', 'Dutch', 'Swedish', 'Polish']
    
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Spain', 'Italy', 'Japan', 'Australia', 'Brazil',
               'India', 'China', 'Mexico', 'South Korea', 'Russia', 'South Africa', 'Argentina', 'Netherlands',
               'Sweden', 'Singapore', 'UAE', 'Switzerland', 'Norway', 'New Zealand', 'Ireland']
    
    loyalty_tiers = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
    loyalty_weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # Higher weights for lower tiers
    
    users = []
    for user_id in range(1, num_users + 1):
        # Generate a signup date between 1-5 years ago
        signup_days_ago = random.randint(30, 5*365)
        signup_date = (datetime.now() - pd.Timedelta(days=signup_days_ago)).strftime('%Y-%m-%d')
        
        # Generate a birth date for someone 18-80 years old
        age = random.randint(18, 80)
        birth_year = datetime.now().year - age
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, 28)
        dob = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"
        
        # Generate weighted loyalty tier (more lower tiers)
        loyalty_tier = random.choices(loyalty_tiers, weights=loyalty_weights)[0]
        
        # Generate email domain based on user ID to ensure uniqueness
        email_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com']
        first = random.choice(first_names).lower()
        last = random.choice(last_names).lower()
        
        # Email generation strategy varies to simulate different patterns
        email_type = random.randint(1, 5)
        if email_type == 1:
            email = f"{first}.{last}@{random.choice(email_domains)}"
        elif email_type == 2:
            email = f"{first}{last[0]}@{random.choice(email_domains)}"
        elif email_type == 3:
            email = f"{first[0]}{last}@{random.choice(email_domains)}"
        elif email_type == 4:
            email = f"{first}{random.randint(1, 999)}@{random.choice(email_domains)}"
        else:
            email = f"{last}.{first[0]}@{random.choice(email_domains)}"
        
        users.append({
            'user_id': user_id,
            'first_name': random.choice(first_names),
            'last_name': random.choice(last_names),
            'email': email,
            'date_of_birth': dob,
            'signup_date': signup_date,
            'preferred_language': random.choice(languages),
            'country_of_residence': random.choice(countries),
            'loyalty_tier': loyalty_tier
        })
    
    users_df = pd.DataFrame(users)
    
    # Save the data
    users_df.to_csv('api_data/users_api.csv', index=False)
    
    print(f"Generated {len(users_df)} synthetic user profiles.")
    
    return users_df

def generate_user_preferences(users_df):
    """
    Generate user preferences based on user data
    
    Args:
        users_df: DataFrame with user data
        
    Returns:
        DataFrame with user preferences
    """
    print(f"Generating user preferences for {len(users_df)} users...")
    
    all_activities = ['Adventure', 'Cultural', 'Culinary', 'Historical', 'Nature', 
                     'Relaxation', 'Religious', 'Wildlife', 'Shopping', 'Nightlife', 
                     'Photography', 'Arts', 'Sports', 'Water Activities', 'Hiking']
    
    all_climates = ['Tropical', 'Mediterranean', 'Desert', 'Continental', 
                   'Temperate', 'Arctic', 'Alpine']
    
    preferences = []
    
    for _, user in users_df.iterrows():
        # Each user likes multiple activities (1-5)
        num_activities = random.randint(1, 5)
        preferred_activities = random.sample(all_activities, num_activities)
        
        # Each user likes multiple climates (1-3)
        num_climates = random.randint(1, 3)
        preferred_climates = random.sample(all_climates, num_climates)
        
        # Budget preferences skew with loyalty tier
        budget_categories = ['Budget', 'Moderate', 'Luxury']
        budget_weights = [0.33, 0.33, 0.34]  # Default equal weights
        
        if user['loyalty_tier'] == 'Bronze':
            budget_weights = [0.6, 0.3, 0.1]
        elif user['loyalty_tier'] == 'Silver':
            budget_weights = [0.4, 0.5, 0.1]
        elif user['loyalty_tier'] == 'Gold':
            budget_weights = [0.2, 0.5, 0.3]
        elif user['loyalty_tier'] == 'Platinum':
            budget_weights = [0.1, 0.4, 0.5]
        elif user['loyalty_tier'] == 'Diamond':
            budget_weights = [0.05, 0.25, 0.7]
        
        preferred_budget = random.choices(budget_categories, weights=budget_weights)[0]
        
        # Preferred star rating correlates with budget
        if preferred_budget == 'Budget':
            preferred_stars = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.15, 0.05])[0]
        elif preferred_budget == 'Moderate':
            preferred_stars = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.4, 0.4, 0.05])[0]
        else:  # Luxury
            preferred_stars = random.choices([1, 2, 3, 4, 5], weights=[0.01, 0.04, 0.15, 0.3, 0.5])[0]
        
        # Travel style
        travel_styles = ['Solo', 'Couple', 'Family', 'Group']
        
        # Age correlates with travel style
        age = datetime.now().year - int(user['date_of_birth'].split('-')[0])
        
        if age < 30:
            style_weights = [0.4, 0.3, 0.1, 0.2]  # More solo/couple for young people
        elif 30 <= age < 40:
            style_weights = [0.2, 0.4, 0.3, 0.1]  # More couple/family for 30s
        elif 40 <= age < 60:
            style_weights = [0.1, 0.2, 0.5, 0.2]  # More family for 40-60
        else:
            style_weights = [0.2, 0.4, 0.1, 0.3]  # More couple/group for seniors
            
        travel_style = random.choices(travel_styles, weights=style_weights)[0]
        
        # Adjust travel style if from certain countries (family-oriented cultures)
        family_oriented_countries = ['India', 'China', 'Mexico', 'Italy', 'Brazil']
        if user['country_of_residence'] in family_oriented_countries and random.random() < 0.7:
            travel_style = 'Family'
        
        preferences.append({
            'preference_id': user['user_id'],
            'user_id': user['user_id'],
            'preferred_hotel_stars': preferred_stars,
            'preferred_budget_category': preferred_budget,
            'preferred_activities': json.dumps(preferred_activities),
            'preferred_climates': json.dumps(preferred_climates),
            'travel_style': travel_style,
            'maximum_flight_duration': random.choice([4, 8, 12, 24, 48])
        })
    
    preferences_df = pd.DataFrame(preferences)
    
    # Save the data
    preferences_df.to_csv('api_data/user_preferences_api.csv', index=False)
    
    print(f"Generated preferences for {len(preferences_df)} users.")
    
    return preferences_df

def fetch_amadeus_data(limit=1000, api_key=None, api_secret=None):
    """
    Fetch travel data from Amadeus API - a more comprehensive travel API
    
    Args:
        limit: Maximum number of destinations to retrieve
        api_key: Amadeus API key
        api_secret: Amadeus API secret
    
    Returns:
        DataFrame with destination and hotel data
    """
    try:
        if api_key is None or api_secret is None:
            # Try to get API key from environment variables
            api_key = os.environ.get('AMADEUS_API_KEY')
            api_secret = os.environ.get('AMADEUS_API_SECRET')
            
            if api_key is None or api_secret is None:
                print("Warning: Amadeus API credentials not provided. Skipping Amadeus API.")
                return None
        
        print("Fetching data from Amadeus API...")
        
        # First, authenticate with Amadeus to get access token
        auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": api_secret
        }
        
        response = requests.post(auth_url, data=auth_data)
        if response.status_code != 200:
            print(f"Authentication failed: {response.text}")
            return None
            
        token_data = response.json()
        access_token = token_data.get('access_token')
        
        if not access_token:
            print("Failed to obtain access token.")
            return None
        
        # Use Amadeus endpoints to fetch travel data
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        # You can use various Amadeus endpoints to fetch different types of data:
        # - https://test.api.amadeus.com/v1/reference-data/locations/cities - City data
        # - https://test.api.amadeus.com/v2/shopping/hotel-offers - Hotel offers
        # - https://test.api.amadeus.com/v1/reference-data/locations/pois - Points of interest
        
        # Example: Fetch cities as destinations
        cities_url = "https://test.api.amadeus.com/v1/reference-data/locations/cities"
        
        all_destinations = []
        destination_id = 1
        
        # Process each continent/region
        for country_code in ["US", "FR", "ES", "IT", "DE", "JP", "AU", "ZA", "BR"]:
            params = {
                "countryCode": country_code,
                "max": 20  # Get up to 20 cities per country
            }
            
            response = requests.get(cities_url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"Error fetching cities for {country_code}: {response.text}")
                continue
                
            cities_data = response.json().get('data', [])
            
            for city in cities_data:
                # Extract city data
                destination = {
                    'destination_id': destination_id,
                    'name': city.get('name', f"Destination {destination_id}"),
                    'country': city.get('countryCode', country_code),
                    'region': city.get('stateCode', 'Unknown'),
                    'latitude': city.get('geoCode', {}).get('latitude', 0),
                    'longitude': city.get('geoCode', {}).get('longitude', 0),
                    'popularity_score': random.uniform(1, 10),
                    'average_rating': random.uniform(3, 5),
                    'climate_type': assign_climate_type(city.get('geoCode', {}).get('latitude', 0)),
                    'best_season_to_visit': assign_best_season(city.get('geoCode', {}).get('latitude', 0)),
                    'description': f"Beautiful destination in {city.get('name', 'this country')}.",
                    'image_url': ''
                }
                
                all_destinations.append(destination)
                destination_id += 1
                
                # If we've reached the limit, stop
                if destination_id > limit:
                    break
            
            # If we've reached the limit, stop
            if destination_id > limit:
                break
            
            # Add a delay between countries to avoid rate limiting
            time.sleep(1)
            
        # Convert to DataFrame
        destinations_df = pd.DataFrame(all_destinations)
        
        # Save the raw API data
        if not destinations_df.empty:
            destinations_df.to_csv('api_data/amadeus_destinations.csv', index=False)
            print(f"Fetched and saved {len(destinations_df)} destinations from Amadeus.")
            return destinations_df
        else:
            print("No destinations found from Amadeus API.")
            return None
            
    except Exception as e:
        print(f"Error fetching data from Amadeus API: {e}")
        return None

def fetch_data():
    """
    Main function to fetch data from APIs and insert into database.
    Will clear existing data before inserting new data.
    Falls back to OpenTripMap if Amadeus fails.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Step 1: Clear existing data
        if not clear_existing_data():
            print("Failed to clear existing data. API fetch aborted.")
            return False
        
        # Step 2: Try to fetch from Amadeus API first (preferred source)
        print("Attempting to fetch data from Amadeus API (preferred source)...")
        amadeus_data = fetch_amadeus_data()
        
        if amadeus_data is not None and not amadeus_data.empty:
            print("Successfully fetched data from Amadeus API.")
            destinations_df = amadeus_data
        else:
            # Step 3: Fall back to OpenTripMap if Amadeus fails
            print("Falling back to OpenTripMap API...")
            destinations_df = fetch_opentrip_map_data(limit=200)
            
            if destinations_df is None or destinations_df.empty:
                print("Failed to fetch data from any API.")
                return False
        
        # Step 4: Fetch or generate hotel data
        print("Fetching hotel data...")
        hotels_df = fetch_hotel_data(destinations_df)
        
        # Step 5: Generate user data and preferences
        print("Generating user data...")
        users_df = generate_synthetic_user_data(1000)
        
        print("Generating user preferences...")
        preferences_df = generate_user_preferences(users_df)
        
        # Step 6: Insert all data into database using batch inserts
        print("Inserting data into database...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Use batch inserts to improve performance
        batch_size = 50  # Insert 50 records at a time
        
        # Insert Destinations in batches
        print("Inserting destinations...")
        for i in range(0, len(destinations_df), batch_size):
            batch = destinations_df.iloc[i:i+batch_size]
            
            # Prepare the batch insertion statements
            for _, dest in batch.iterrows():
                # Add image_url if missing
                image_url = dest.get('image_url', '')
                
                cursor.execute("""
                    INSERT INTO destinations 
                    (destination_id, name, country, region, latitude, longitude, 
                    popularity_score, average_rating, climate_type, best_season_to_visit, description, image_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, 
                dest['destination_id'], dest['name'], dest['country'], dest['region'], 
                dest['latitude'], dest['longitude'], dest['popularity_score'], 
                dest['average_rating'], dest['climate_type'], dest['best_season_to_visit'], 
                dest['description'], image_url)
            
            # Commit after each batch
            conn.commit()
            print(f"Inserted destinations batch {i//batch_size + 1}/{(len(destinations_df)-1)//batch_size + 1}")
        
        # Insert Hotels in batches
        print("Inserting hotels...")
        for i in range(0, len(hotels_df), batch_size):
            batch = hotels_df.iloc[i:i+batch_size]
            
            for _, hotel in batch.iterrows():
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
            
            conn.commit()
            print(f"Inserted hotels batch {i//batch_size + 1}/{(len(hotels_df)-1)//batch_size + 1}")
        
        # Insert Users in batches
        print("Inserting users...")
        for i in range(0, len(users_df), batch_size):
            batch = users_df.iloc[i:i+batch_size]
            
            for _, user in batch.iterrows():
                cursor.execute("""
                    INSERT INTO users 
                    (user_id, first_name, last_name, email, date_of_birth, signup_date, 
                    preferred_language, country_of_residence, loyalty_tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, 
                user['user_id'], user['first_name'], user['last_name'], user['email'], 
                user['date_of_birth'], user['signup_date'], user['preferred_language'], 
                user['country_of_residence'], user['loyalty_tier'])
            
            conn.commit()
            print(f"Inserted users batch {i//batch_size + 1}/{(len(users_df)-1)//batch_size + 1}")
        
        # Insert User Preferences in batches
        print("Inserting user preferences...")
        for i in range(0, len(preferences_df), batch_size):
            batch = preferences_df.iloc[i:i+batch_size]
            
            for _, pref in batch.iterrows():
                cursor.execute("""
                    INSERT INTO user_preferences 
                    (preference_id, user_id, preferred_hotel_stars, preferred_budget_category, 
                    preferred_activities, preferred_climates, travel_style, maximum_flight_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, 
                pref['preference_id'], pref['user_id'], pref['preferred_hotel_stars'], 
                pref['preferred_budget_category'], pref['preferred_activities'], 
                pref['preferred_climates'], pref['travel_style'], pref['maximum_flight_duration'])
            
            conn.commit()
            print(f"Inserted preferences batch {i//batch_size + 1}/{(len(preferences_df)-1)//batch_size + 1}")
        
        conn.close()
        
        print("API data fetch and database insertion complete!")
        return True
        
    except Exception as e:
        print(f"Error in fetch_data: {e}")
        return False

def main():
    """
    Main function to fetch and generate all data
    """
    print("Fetching destination data from public APIs...")
    
    # Call fetch_data with database insertion
    success = fetch_data()
    
    if success:
        print("Data fetching and insertion complete!")
    else:
        print("Data fetching and insertion failed.")

if __name__ == "__main__":
    main() 