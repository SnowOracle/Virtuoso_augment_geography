# Database Schema Design

## Users Table
- user_id (PK)
- first_name
- last_name
- email
- date_of_birth
- signup_date
- preferred_language
- country_of_residence
- loyalty_tier

## Destinations Table
- destination_id (PK)
- name
- country
- region
- latitude
- longitude
- popularity_score
- average_rating
- climate_type
- best_season_to_visit
- description

## Hotels Table
- hotel_id (PK)
- name
- destination_id (FK)
- star_rating
- price_category
- has_pool
- has_spa
- has_restaurant
- rooms_available
- average_rating
- address
- latitude
- longitude

## Tours Table
- tour_id (PK)
- name
- destination_id (FK)
- duration_hours
- price
- category (adventure, cultural, culinary, etc.)
- group_size_limit
- difficulty_level
- average_rating
- description
- availability_schedule

## Bookings Table
- booking_id (PK)
- user_id (FK)
- booking_date
- total_cost
- payment_status
- booking_status (confirmed, canceled, completed)
- booking_channel (web, mobile, agent)

## HotelBookings Table
- hotel_booking_id (PK)
- booking_id (FK)
- hotel_id (FK)
- check_in_date
- check_out_date
- room_type
- number_of_guests
- special_requests
- rate_per_night

## TourBookings Table
- tour_booking_id (PK)
- booking_id (FK)
- tour_id (FK)
- tour_date
- number_of_participants
- special_requirements

## UserPreferences Table
- preference_id (PK)
- user_id (FK)
- preferred_hotel_stars
- preferred_budget_category
- preferred_activities (json/array)
- preferred_climates (json/array)
- travel_style
- maximum_flight_duration

## UserInteractions Table
- interaction_id (PK)
- user_id (FK)
- interaction_type (search, view, save, rate)
- entity_type (hotel, tour, destination)
- entity_id
- timestamp
- interaction_details (search terms, rating value, etc.)
- session_id

## Reviews Table
- review_id (PK)
- user_id (FK)
- entity_type (hotel, tour)
- entity_id
- rating
- comment
- review_date
- helpful_votes 