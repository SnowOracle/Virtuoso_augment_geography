# Data Directory

This directory stores data files for the Virtuoso Travel Recommendation System.

## Sample Data Files

Sample data files are included for reference and testing:

- `sample_users.csv` - Sample user data
- `sample_destinations.csv` - Sample destination data
- `sample_hotels.csv` - Sample hotel data
- `sample_tours.csv` - Sample tour data

## Database Initialization

When setting up the application for the first time:

1. Enable Admin Mode in the sidebar
2. Go to the Database Setup tab
3. Click "Run Complete Setup"

This will initialize the database schema and populate it with sample data.

## Data Sources

The application can use data from multiple sources:

1. Sample data included in this directory
2. Generated simulation data (created by data_generator.py)
3. External API data (fetched by api_data_fetcher.py if configured)

## Custom Data

To use your own data, replace the sample CSV files with your own data following the same schema.
The database_setup.py script will import your custom data during setup. 