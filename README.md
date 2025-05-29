# Virtuoso Travel Recommendation System

A sophisticated travel recommendation system built with Streamlit, offering personalized destination recommendations based on user preferences, travel history, and collaborative filtering.

![Virtuoso Travel Banner](https://source.unsplash.com/featured/1200x400/?travel)

## 🌟 Features

- **Personalized Recommendations**: Get destination suggestions tailored to your travel preferences and history
- **Interactive Maps**: Visualize recommended destinations and your travel history on interactive maps
- **User Profiles**: View detailed user profiles with travel preferences and history
- **Advanced Analytics**: Explore travel statistics and trends
- **Admin Dashboard**: Manage database, users, and system settings

## 📋 Requirements

- Python 3.7+
- SQL Server database
- ODBC driver for SQL Server
- Required Python packages (see `requirements.txt`)

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Virtuoso_augment_geography.git
   cd Virtuoso_augment_geography
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your database connection in `app.py`:
   ```python
   conn_str = "DRIVER=/path/to/your/driver;SERVER=your_server;PORT=1433;DATABASE=virtuoso_travel;UID=your_username;PWD=your_password;TDS_Version=7.4;"
   ```

## 🏃‍♂️ Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Access the web interface at `http://localhost:8501`

3. For first-time setup:
   - Enable Admin Mode from the sidebar
   - Go to the Database Setup tab
   - Click "Run Complete Setup" to initialize the database and load sample data

4. Switch back to user mode to explore recommendations

## 🔧 System Architecture

The recommendation system uses a hybrid approach combining:

- **Collaborative Filtering**: Recommendations based on similar users' preferences
- **Content-Based Filtering**: Matches based on destination attributes and user preferences
- **Feature Engineering**: Advanced feature extraction for better predictions
- **XGBoost Model**: Machine learning model for personalized scoring

## 📊 Data Model

The system relies on several interconnected data entities:

- Users and user preferences
- Destinations with climate and regional data
- Hotels and tours
- Bookings and travel history
- User interactions and reviews

## 📸 Screenshots

*Add screenshots of your application here*

## 🧪 Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Your Name - *Initial work*

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) - For the amazing web framework
- [Folium](https://python-visualization.github.io/folium/) - For interactive maps
- [XGBoost](https://xgboost.readthedocs.io/) - For the recommendation model
- [Plotly](https://plotly.com/) - For beautiful visualizations 