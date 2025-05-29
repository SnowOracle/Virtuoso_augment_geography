-- Setup for Virtuoso Travel Database
-- Run this in SQL Server Management Studio or Azure Data Studio

-- 1. Create Database
USE master;
GO

IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'virtuoso_travel')
BEGIN
    CREATE DATABASE virtuoso_travel;
END
GO

USE virtuoso_travel;
GO

-- 2. Create Login and User
-- First create the SQL Server login
IF NOT EXISTS (SELECT name FROM master.sys.server_principals WHERE name = 'climbing_user')
BEGIN
    CREATE LOGIN climbing_user WITH PASSWORD = 'hoosierheights';
END
GO

-- Then create the database user mapped to that login
IF NOT EXISTS (SELECT name FROM sys.database_principals WHERE name = 'climbing_user')
BEGIN
    CREATE USER climbing_user FOR LOGIN climbing_user;
    
    -- Grant permissions
    EXEC sp_addrolemember 'db_datareader', 'climbing_user';
    EXEC sp_addrolemember 'db_datawriter', 'climbing_user';
    EXEC sp_addrolemember 'db_ddladmin', 'climbing_user';
END
GO

-- 3. Create Tables

-- Users Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'users')
BEGIN
    CREATE TABLE users (
        user_id INT PRIMARY KEY,
        first_name NVARCHAR(50) NOT NULL,
        last_name NVARCHAR(50) NOT NULL,
        email NVARCHAR(100) NOT NULL,
        date_of_birth DATE NOT NULL,
        signup_date DATE NOT NULL,
        preferred_language NVARCHAR(30),
        country_of_residence NVARCHAR(50),
        loyalty_tier NVARCHAR(20)
    );
END
GO

-- Destinations Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'destinations')
BEGIN
    CREATE TABLE destinations (
        destination_id INT PRIMARY KEY,
        name NVARCHAR(100) NOT NULL,
        country NVARCHAR(50) NOT NULL,
        region NVARCHAR(50),
        latitude FLOAT NOT NULL,
        longitude FLOAT NOT NULL,
        popularity_score FLOAT,
        average_rating FLOAT,
        climate_type NVARCHAR(30),
        best_season_to_visit NVARCHAR(30),
        description NVARCHAR(MAX),
        image_url NVARCHAR(255)
    );
END
GO

-- Hotels Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'hotels')
BEGIN
    CREATE TABLE hotels (
        hotel_id INT PRIMARY KEY,
        name NVARCHAR(100) NOT NULL,
        destination_id INT NOT NULL,
        star_rating INT NOT NULL,
        price_category NVARCHAR(20) NOT NULL,
        has_pool BIT NOT NULL DEFAULT 0,
        has_spa BIT NOT NULL DEFAULT 0,
        has_restaurant BIT NOT NULL DEFAULT 0,
        rooms_available INT NOT NULL,
        average_rating FLOAT,
        address NVARCHAR(200),
        latitude FLOAT,
        longitude FLOAT,
        FOREIGN KEY (destination_id) REFERENCES destinations(destination_id)
    );
END
GO

-- Tours Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'tours')
BEGIN
    CREATE TABLE tours (
        tour_id INT PRIMARY KEY,
        name NVARCHAR(100) NOT NULL,
        destination_id INT NOT NULL,
        duration_hours FLOAT NOT NULL,
        price DECIMAL(10, 2) NOT NULL,
        category NVARCHAR(30),
        group_size_limit INT,
        difficulty_level NVARCHAR(20),
        average_rating FLOAT,
        description NVARCHAR(MAX),
        availability_schedule NVARCHAR(MAX),
        FOREIGN KEY (destination_id) REFERENCES destinations(destination_id)
    );
END
GO

-- Bookings Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'bookings')
BEGIN
    CREATE TABLE bookings (
        booking_id INT PRIMARY KEY,
        user_id INT NOT NULL,
        booking_date DATE NOT NULL,
        total_cost DECIMAL(10, 2) NOT NULL,
        payment_status NVARCHAR(20) NOT NULL,
        booking_status NVARCHAR(20) NOT NULL,
        booking_channel NVARCHAR(20),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
END
GO

-- Hotel Bookings Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'hotel_bookings')
BEGIN
    CREATE TABLE hotel_bookings (
        hotel_booking_id INT PRIMARY KEY,
        booking_id INT NOT NULL,
        hotel_id INT NOT NULL,
        check_in_date DATE NOT NULL,
        check_out_date DATE NOT NULL,
        room_type NVARCHAR(30) NOT NULL,
        number_of_guests INT NOT NULL,
        special_requests NVARCHAR(MAX),
        rate_per_night DECIMAL(10, 2) NOT NULL,
        FOREIGN KEY (booking_id) REFERENCES bookings(booking_id),
        FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
    );
END
GO

-- Tour Bookings Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'tour_bookings')
BEGIN
    CREATE TABLE tour_bookings (
        tour_booking_id INT PRIMARY KEY,
        booking_id INT NOT NULL,
        tour_id INT NOT NULL,
        tour_date DATE NOT NULL,
        number_of_participants INT NOT NULL,
        special_requirements NVARCHAR(MAX),
        FOREIGN KEY (booking_id) REFERENCES bookings(booking_id),
        FOREIGN KEY (tour_id) REFERENCES tours(tour_id)
    );
END
GO

-- User Preferences Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'user_preferences')
BEGIN
    CREATE TABLE user_preferences (
        preference_id INT PRIMARY KEY,
        user_id INT NOT NULL,
        preferred_hotel_stars INT,
        preferred_budget_category NVARCHAR(20),
        preferred_activities NVARCHAR(MAX), -- Will be stored as JSON
        preferred_climates NVARCHAR(MAX), -- Will be stored as JSON
        travel_style NVARCHAR(30),
        maximum_flight_duration INT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
END
GO

-- User Interactions Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'user_interactions')
BEGIN
    CREATE TABLE user_interactions (
        interaction_id INT PRIMARY KEY,
        user_id INT NOT NULL,
        interaction_type NVARCHAR(30) NOT NULL,
        entity_type NVARCHAR(30) NOT NULL,
        entity_id INT NOT NULL,
        timestamp DATETIME NOT NULL,
        interaction_details NVARCHAR(MAX),
        session_id NVARCHAR(100),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
END
GO

-- Reviews Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'reviews')
BEGIN
    CREATE TABLE reviews (
        review_id INT PRIMARY KEY,
        user_id INT NOT NULL,
        entity_type NVARCHAR(30) NOT NULL,
        entity_id INT NOT NULL,
        rating INT NOT NULL,
        comment NVARCHAR(MAX),
        review_date DATE NOT NULL,
        helpful_votes INT DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
END
GO

-- Create indexes for better performance
CREATE INDEX idx_bookings_user_id ON bookings(user_id);
CREATE INDEX idx_hotels_destination_id ON hotels(destination_id);
CREATE INDEX idx_tours_destination_id ON tours(destination_id);
CREATE INDEX idx_hotel_bookings_booking_id ON hotel_bookings(booking_id);
CREATE INDEX idx_tour_bookings_booking_id ON tour_bookings(booking_id);
CREATE INDEX idx_user_interactions_user_id ON user_interactions(user_id);
CREATE INDEX idx_reviews_user_id ON reviews(user_id);
GO

PRINT 'Database setup completed successfully.'; 