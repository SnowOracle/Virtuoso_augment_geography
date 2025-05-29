# Database Table Columns Reference

## Tables Overview

| Table Name | Description |
|------------|-------------|
| `bookings` | Stores customer booking information |
| `destinations` | Contains travel destinations with geographic data |
| `hotel_bookings` | Hotel-specific booking details |
| `hotels` | Information about available hotels |
| `reviews` | User reviews for destinations, hotels, and tours |
| `tour_bookings` | Tour-specific booking details |
| `tours` | Information about available tours |
| `user_interactions` | Records of user actions (views, likes, searches) |
| `user_preferences` | User travel preferences and settings |
| `users` | User profile and account information |

## Table Columns

### `bookings`

| Column Name | Description |
|-------------|-------------|
| `booking_id` | Primary key for bookings |
| `user_id` | Foreign key to users table |
| `booking_date` | Date when booking was made |
| `total_cost` | Total cost of the booking |
| `payment_status` | Status of payment |
| `booking_status` | Current status (confirmed, canceled, completed) |
| `booking_channel` | Channel through which booking was made (web, mobile, agent) |

### `destinations`

| Column Name | Description |
|-------------|-------------|
| `destination_id` | Primary key for destinations |
| `name` | Destination name |
| `country` | Country where destination is located |
| `region` | Region or area within country |
| `latitude` | Geographic latitude coordinate |
| `longitude` | Geographic longitude coordinate |
| `popularity_score` | Numeric score of destination popularity |
| `average_rating` | Average user rating |
| `climate_type` | Type of climate (tropical, desert, etc.) |
| `best_season_to_visit` | Recommended time of year to visit |
| `description` | Detailed description of destination |
| `image_url` | URL to destination image |

### `hotel_bookings`

| Column Name | Description |
|-------------|-------------|
| `hotel_booking_id` | Primary key for hotel bookings |
| `booking_id` | Foreign key to bookings table |
| `hotel_id` | Foreign key to hotels table |
| `check_in_date` | Date of guest check-in |
| `check_out_date` | Date of guest check-out |
| `room_type` | Type of room booked |
| `number_of_guests` | Number of guests in booking |
| `special_requests` | Any special accommodation requests |
| `rate_per_night` | Cost per night |

### `hotels`

| Column Name | Description |
|-------------|-------------|
| `hotel_id` | Primary key for hotels |
| `name` | Hotel name |
| `destination_id` | Foreign key to destinations table |
| `star_rating` | Hotel quality rating (1-5 stars) |
| `price_category` | Price tier (budget, moderate, luxury) |
| `has_pool` | Whether hotel has a pool |
| `has_spa` | Whether hotel has a spa |
| `has_restaurant` | Whether hotel has a restaurant |
| `rooms_available` | Number of available rooms |
| `average_rating` | Average user rating |
| `address` | Physical address |
| `latitude` | Geographic latitude coordinate |
| `longitude` | Geographic longitude coordinate |

### `reviews`

| Column Name | Description |
|-------------|-------------|
| `review_id` | Primary key for reviews |
| `user_id` | Foreign key to users table |
| `entity_type` | Type of entity being reviewed (hotel, tour, destination) |
| `entity_id` | ID of entity being reviewed |
| `rating` | Numeric rating (usually 1-5) |
| `comment` | Review text |
| `review_date` | Date review was submitted |
| `helpful_votes` | Number of users who found review helpful |

### `tour_bookings`

| Column Name | Description |
|-------------|-------------|
| `tour_booking_id` | Primary key for tour bookings |
| `booking_id` | Foreign key to bookings table |
| `tour_id` | Foreign key to tours table |
| `tour_date` | Date of scheduled tour |
| `number_of_participants` | Number of people in tour booking |
| `special_requirements` | Any special requests for tour |

### `tours`

| Column Name | Description |
|-------------|-------------|
| `tour_id` | Primary key for tours |
| `name` | Tour name |
| `destination_id` | Foreign key to destinations table |
| `duration_hours` | Length of tour in hours |
| `price` | Tour price |
| `category` | Type of tour (adventure, cultural, culinary, etc.) |
| `group_size_limit` | Maximum number of participants |
| `difficulty_level` | Physical difficulty rating |
| `average_rating` | Average user rating |
| `description` | Detailed description of tour |
| `availability_schedule` | Schedule of when tour is available |

### `user_interactions`

| Column Name | Description |
|-------------|-------------|
| `interaction_id` | Primary key for interactions |
| `user_id` | Foreign key to users table |
| `interaction_type` | Type of interaction (search, view, save, rate) |
| `entity_type` | Type of entity interacted with |
| `entity_id` | ID of entity interacted with |
| `timestamp` | When interaction occurred |
| `interaction_details` | Additional details (search terms, etc.) |
| `session_id` | User session identifier |

### `user_preferences`

| Column Name | Description |
|-------------|-------------|
| `preference_id` | Primary key for preferences |
| `user_id` | Foreign key to users table |
| `preferred_hotel_stars` | Preferred hotel quality level |
| `preferred_budget_category` | Budget preference |
| `preferred_activities` | List of preferred activities (JSON array) |
| `preferred_climates` | List of preferred climate types (JSON array) |
| `travel_style` | General travel style preference |
| `maximum_flight_duration` | Maximum acceptable flight time |

### `users`

| Column Name | Description |
|-------------|-------------|
| `user_id` | Primary key for users |
| `first_name` | User's first name |
| `last_name` | User's last name |
| `email` | User's email address |
| `date_of_birth` | User's birth date |
| `signup_date` | Date user registered |
| `preferred_language` | User's language preference |
| `country_of_residence` | User's home country |
| `loyalty_tier` | Loyalty program tier status | 