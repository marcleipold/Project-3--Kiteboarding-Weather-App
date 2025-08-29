import streamlit as st

st.set_page_config(
    page_title="🏄‍♂️ Kiteboarding Weather Predictor - EXTREVITY.COM",
    layout="centered"  # or "wide"
)

# Import Dependencies
import numpy as np
import datetime
import requests
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO
import base64
import pytz
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import os
from dotenv import load_dotenv
import gzip
import json


# Load environment variables
load_dotenv()

# Access Keys & Credentials
owm_api = os.environ.get('OPENWEATHERMAP_API_KEY')

if not owm_api:
    st.error("⚠️ OpenWeatherMap API key not found. Please add it to your .env file.")
    st.stop()

# Initialize session state for data persistence
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None

# Load wind chart data
@st.cache_data
def load_wind_chart():
    try:
        wind_chart = pd.read_csv("wind_chart_df.csv", index_col=[0])
        wind_chart_df = pd.DataFrame(wind_chart)
        wind_chart_df.columns = wind_chart_df.columns.astype("int64")
        return wind_chart_df
    except FileNotFoundError:
        st.warning("⚠️ wind_chart_df.csv not found. Using sample data.")
        # Create sample wind chart data
        weights = range(40, 100, 5)
        winds = range(10, 41)
        data = {}
        for wind in winds:
            data[wind] = []
            for weight in weights:
                # Simple formula for kite size based on weight and wind
                kite_size = max(3, min(19, int(20 - (wind * 0.3) + (weight - 70) * 0.1)))
                data[wind].append(kite_size)
        wind_chart_df = pd.DataFrame(data, index=weights)
        return wind_chart_df

wind_chart_df = load_wind_chart()

# Load city list
@st.cache_data
def load_city_list():
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        city_list_file = os.path.join(current_dir, "city.list.json.gz")
        
        with gzip.open(city_list_file, "rt", encoding="utf-8") as f:
            city_data = json.load(f)
        
        city_names = [city['name'] for city in city_data]
        return city_names
    except (FileNotFoundError, json.JSONDecodeError):
        st.info("ℹ️ City list file not found. Using default cities.")
        return ["London", "New York", "Paris", "Tokyo", "Sydney", "Miami", "San Francisco", 
                "Barcelona", "Cape Town", "Rio de Janeiro", "Dubai", "Hawaii"]

city_names = load_city_list()

# Custom CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Ubuntu&display=swap" rel="stylesheet">
<style>
html, body, .main {
    background-color: #ED1C24 !important;
    color: black !important;
    font-family: 'Ubuntu', sans-serif !important;
}

section[data-testid="stSidebar"],
div[data-testid="stVerticalBlock"] {
    background-color: white !important;
}

label, .stRadio, .stSelectbox, .stTextInput, .stMultiselect, .stNumberInput {
    color: black !important;
}

/* Existing styles you already have */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #ed1c24;
    transform: scale(1.05);
}
.forecast-table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}
.forecast-table th, .forecast-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
}
.forecast-table th {
    background-color: #f2f2f2;
    font-weight: bold;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 20px;
    color: white;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏄‍♂️ Kiteboarding Weather Predictor - EXTREVITY.COM")
st.markdown("### Your personal weather assistant for perfect kiteboarding conditions")

# User inputs in an expander for cleaner UI
with st.expander("⚙️ Configure Your Settings", expanded=True):
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # City selection
        default_city_index = city_names.index("Chicago") if "Chicago" in city_names else 0
        city = st.selectbox(
            "📍 Select City",
            city_names,
            index=default_city_index,
            help="Choose your kiteboarding location"
        )
    
    with col2:
        # Weight input
        weight_val = st.number_input(
            "⚖️ Your Weight",
            min_value=30,
            max_value=150,
            value=75,
            step=5,
            help="Enter your weight for kite size recommendations"
        )
        weight_unit_val = st.radio("Unit", ["kg", "lbs"], horizontal=True)
    
    with col3:
        # Kite sizes
        kite_sizes = [f"{i}m" for i in range(3, 20)]
        selected_kite_sizes = st.multiselect(
            "🪁 Your Kite Sizes",
            kite_sizes,
            default=["9m", "12m"],
            help="Select the kite sizes you own"
        )

    col4, col5 = st.columns(2)
    with col4:
        unit = st.radio("🌡️ Temperature", ["Celsius", "Fahrenheit"], horizontal=True)
    with col5:
        speed = st.radio("💨 Wind Speed", ["Knots", "km/h", "mph"], horizontal=True, index=0)

# Convert units
weight_in_kg = weight_val if weight_unit_val == "kg" else round(weight_val * 0.453592)
selected_kite_sizes_int = [int(size[:-1]) for size in selected_kite_sizes]

# Helper functions
def kelvin_to_temp(kelvin, unit="Celsius"):
    """Convert Kelvin to Celsius or Fahrenheit"""
    celsius = kelvin - 273.15
    if unit == "Celsius":
        return round(celsius)
    else:
        return round(celsius * 1.8 + 32)

def convert_wind_speed(speed_ms, unit="Knots"):
    """Convert wind speed from m/s to desired unit"""
    if unit == "Knots":
        return round(speed_ms * 1.94384)
    elif unit == "km/h":
        return round(speed_ms * 3.6)
    else:  # mph
        return round(speed_ms * 2.23694)

def get_wind_unit_symbol(unit):
    """Get the symbol for wind unit"""
    if unit == "Knots":
        return "kt"
    elif unit == "km/h":
        return "km/h"
    else:
        return "mph"

def get_kite_size_recommendation(weight_kg, wind_speed_kt):
    """Get recommended kite size based on weight and wind speed"""
    try:
        # Convert wind speed to knots if needed
        wind_col = min(40, max(10, int(wind_speed_kt)))
        
        # Find closest weight in index
        available_weights = wind_chart_df.index.tolist()
        closest_weight = min(available_weights, key=lambda x: abs(x - weight_kg))
        
        if wind_col in wind_chart_df.columns:
            return int(wind_chart_df.loc[closest_weight, wind_col])
    except:
        pass
    
    # Fallback formula if chart lookup fails
    return max(3, min(19, int(20 - (wind_speed_kt * 0.3) + (weight_kg - 70) * 0.1)))

def create_wind_arrow(direction):
    """Create a wind direction arrow"""
    # Create a simple arrow using matplotlib
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.arrow(0.5, 0.5, 0.3 * np.sin(np.radians(direction)), 
             0.3 * np.cos(np.radians(direction)),
             head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode()

# Main forecast button
if st.button("🔍 Get Forecast", type="primary", use_container_width=True):
    with st.spinner("Fetching weather data..."):
        try:
            # Get current weather
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={owm_api}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                current_data = response.json()
                st.session_state.weather_data = current_data
                
                # Get coordinates
                lat = current_data["coord"]["lat"]
                lon = current_data["coord"]["lon"]
                
                # Get forecast data using One Call API 3.0
                forecast_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&appid={owm_api}"
                forecast_response = requests.get(forecast_url, timeout=10)
                
                # If One Call 3.0 doesn't work, try 2.5
                if forecast_response.status_code != 200:
                    forecast_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly&appid={owm_api}"
                    forecast_response = requests.get(forecast_url, timeout=10)
                
                if forecast_response.status_code == 200:
                    st.session_state.forecast_data = forecast_response.json()
                else:
                    # Fallback to 5-day forecast
                    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={owm_api}"
                    forecast_response = requests.get(forecast_url, timeout=10)
                    if forecast_response.status_code == 200:
                        # Process 5-day forecast into daily format
                        raw_forecast = forecast_response.json()
                        daily_data = []
                        current_date = None
                        day_data = []
                        
                        for item in raw_forecast['list']:
                            date = datetime.datetime.fromtimestamp(item['dt']).date()
                            if current_date != date:
                                if day_data:
                                    # Aggregate day data
                                    daily_data.append({
                                        'dt': day_data[0]['dt'],
                                        'temp': {
                                            'max': max([d['main']['temp_max'] for d in day_data]),
                                            'min': min([d['main']['temp_min'] for d in day_data])
                                        },
                                        'wind_speed': np.mean([d['wind']['speed'] for d in day_data]),
                                        'wind_gust': max([d['wind'].get('gust', d['wind']['speed']) for d in day_data]),
                                        'wind_deg': day_data[0]['wind'].get('deg', 0),
                                        'humidity': np.mean([d['main']['humidity'] for d in day_data]),
                                        'clouds': np.mean([d['clouds']['all'] for d in day_data]),
                                        'pop': max([d.get('pop', 0) for d in day_data]),
                                        'weather': day_data[0]['weather']
                                    })
                                current_date = date
                                day_data = [item]
                            else:
                                day_data.append(item)
                        
                        st.session_state.forecast_data = {
                            'daily': daily_data[:8],
                            'timezone_offset': 0
                        }
                    else:
                        st.error(f"Unable to fetch forecast data. Status: {forecast_response.status_code}")
                        st.stop()
            else:
                st.error(f"City not found or API error. Status: {response.status_code}")
                st.stop()
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
            st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.stop()

# Display results if data is available
if st.session_state.weather_data and st.session_state.forecast_data:
    current_data = st.session_state.weather_data
    forecast_data = st.session_state.forecast_data
    
    # Current weather display
    st.markdown("---")
    st.subheader(f"📍 Current Weather in {city.title()}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_temp = kelvin_to_temp(current_data["main"]["temp"], unit)
    current_wind = convert_wind_speed(current_data["wind"]["speed"], speed)
    temp_symbol = "°C" if unit == "Celsius" else "°F"
    wind_symbol = get_wind_unit_symbol(speed)
    
    with col1:
        st.metric("🌡️ Temperature", f"{current_temp}{temp_symbol}")
    with col2:
        st.metric("💨 Wind Speed", f"{current_wind} {wind_symbol}")
    with col3:
        st.metric("💧 Humidity", f"{current_data['main']['humidity']}%")
    with col4:
        st.metric("☁️ Clouds", f"{current_data['clouds']['all']}%")
    
    # Map
    st.markdown("---")
    st.subheader("🗺️ Location Map")
    
    m = folium.Map(location=[current_data["coord"]["lat"], current_data["coord"]["lon"]], 
                   zoom_start=10, height=400)
    folium.Marker(
        [current_data["coord"]["lat"], current_data["coord"]["lon"]],
        popup=f"{city} - {current_data['weather'][0]['description'].title()}",
        tooltip=f"Wind: {current_wind} {wind_symbol}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    
    folium_static(m, width=700, height=400)
    
    # Process forecast data
    st.markdown("---")
    st.subheader("📅 8-Day Forecast")
    
    # Prepare forecast data
    dates = []
    wind_speeds = []
    wind_gusts = []
    wind_directions = []
    temperatures = []
    rain_chances = []
    kite_recommendations = []
    can_kite = []
    
    for day in forecast_data.get('daily', [])[:8]:
        # Date
        dt = datetime.datetime.fromtimestamp(day['dt'])
        dates.append(dt.strftime('%b %d'))
        
        # Wind
        wind_speed_kt = convert_wind_speed(day.get('wind_speed', 0), "Knots")
        wind_speed_display = convert_wind_speed(day.get('wind_speed', 0), speed)
        wind_speeds.append(wind_speed_display)
        wind_gusts.append(convert_wind_speed(day.get('wind_gust', day.get('wind_speed', 0)), speed))
        wind_directions.append(day.get('wind_deg', 0))
        
        # Temperature
        if 'temp' in day and isinstance(day['temp'], dict):
            temp_max = kelvin_to_temp(day['temp']['max'], unit)
        else:
            temp_max = kelvin_to_temp(day.get('temp', 273.15), unit)
        temperatures.append(temp_max)
        
        # Rain
        rain_chances.append(int(day.get('pop', 0) * 100))
        
        # Kite recommendation
        kite_size = get_kite_size_recommendation(weight_in_kg, wind_speed_kt)
        kite_recommendations.append(kite_size)
        
        # Can kite today?
        can_kite_today = any(abs(kite_size - size) <= 2 for size in selected_kite_sizes_int)
        can_kite.append(can_kite_today)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': dates,
        f'Wind ({wind_symbol})': wind_speeds,
        f'Gust ({wind_symbol})': wind_gusts,
        'Direction (°)': wind_directions,
        f'Temp ({temp_symbol})': temperatures,
        'Rain (%)': rain_chances,
        'Kite Size': [f"{k}m" for k in kite_recommendations],
        'Can Kite?': ['✅' if c else '❌' for c in can_kite]
    })
    
    # Display forecast table
    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Can Kite?": st.column_config.TextColumn(
                "Can Kite?",
                help="Based on your available kite sizes"
            ),
        }
    )
    
    # Wind speed chart with kite images
    st.markdown("---")
    st.subheader("📊 Wind Speed Forecast with Kite Recommendations")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(dates, wind_speeds, color='skyblue', edgecolor='navy', linewidth=2)
    
    # Color bars based on wind speed
    for bar, wind, kite in zip(bars, wind_speeds, kite_recommendations):
        if wind < 10:
            bar.set_facecolor('#6286B7')  # Light blue
        elif wind < 15:
            bar.set_facecolor('#4A94A9')  # Teal
        elif wind < 20:
            bar.set_facecolor('#53A553')  # Green
        elif wind < 25:
            bar.set_facecolor('#A79D51')  # Yellow-brown
        elif wind < 30:
            bar.set_facecolor('#A16C5C')  # Brown
        elif wind < 35:
            bar.set_facecolor('#813A4E')  # Dark red
        else:
            bar.set_facecolor('#AF5088')  # Purple
        
        # Try to add kite images from URL or local files
        try:
            # First try to load from URL (your original code)
            url = f'https://extrevity.com/wp-content/uploads/2021/11/{kite}Artboard-1@2x.png'
            response = requests.get(url, timeout=2)
            
            if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('image/'):
                kite_img = Image.open(BytesIO(response.content))
            else:
                # Try local file as fallback
                local_kite_path = f'images/{kite}Artboard-1@2x.png'
                if os.path.exists(local_kite_path):
                    kite_img = Image.open(local_kite_path)
                else:
                    # Skip image if not found
                    kite_img = None
            
            if kite_img:
                # Add kite image to the bar
                kite_img.thumbnail((50, 25))
                offset_img = OffsetImage(kite_img, zoom=1.0)
                offset_img.image.axes = ax
                
                x_pos = bar.get_x() + bar.get_width() / 2.0
                y_pos = bar.get_y() + bar.get_height() - 1.5
                ab = AnnotationBbox(offset_img, (x_pos, y_pos), xycoords='data', frameon=False)
                ax.add_artist(ab)
        except:
            pass  # Skip if image loading fails
        
        # Always add text label as fallback/addition
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{kite}m', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'Wind Speed ({wind_symbol})', fontsize=12)
    ax.set_title('Wind Speed and Recommended Kite Sizes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', label='Light Wind (<10)'),
        Patch(facecolor='#FFD700', label='Moderate (10-20)'),
        Patch(facecolor='#FFA500', label='Strong (20-30)'),
        Patch(facecolor='#FF6347', label='Very Strong (>30)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Summary
    st.markdown("---")
    st.subheader("📝 Summary")
    
    good_days = sum(can_kite)
    if good_days > 0:
        st.success(f"🎉 You can kitesurf on {good_days} out of {len(dates)} days with your current kite sizes!")
        
        best_day_idx = wind_speeds.index(max(wind_speeds))
        st.info(f"💨 Best wind day: {dates[best_day_idx]} with {wind_speeds[best_day_idx]} {wind_symbol}")
    else:
        st.warning("⚠️ No ideal kitesurfing days with your current kite sizes. Consider getting different sizes!")
        
        recommended_sizes = set(kite_recommendations)
        st.info(f"💡 Recommended kite sizes for this week: {', '.join([f'{k}m' for k in sorted(recommended_sizes)])}")

# Disclaimer
with st.expander("⚖️ Disclaimer"):
    st.markdown("""
    **Weather App Disclaimer**
    
    This app provides weather forecasts for kiteboarding enthusiasts. While we strive to provide accurate information,
    weather conditions can change rapidly. Always:
    
    - Check local conditions before heading out
    - Use proper safety equipment
    - Never kite alone in dangerous conditions
    - Respect local regulations and beach rules
    - Consider your skill level when choosing conditions
    
    The developers assume no responsibility for decisions made based on this app's information.
    Stay safe and have fun! 🪁
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for the kiteboarding community")

# Below your st.title()
if st.button("🔁 Refresh App", help="Clear cache and reload app"):
    st.cache_data.clear()
    st.experimental_rerun()