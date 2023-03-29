import numpy as np
import streamlit as st
import datetime,requests
from plotly import graph_objects as go
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib as mpl
from matplotlib import cm
from colorspacious import cspace_converter

#load in the wind chart and create a dataframe

wind_chart = pd.read_csv("wind_chart_df.csv", index_col=[0])
wind_chart_df = pd.DataFrame(wind_chart)
wind_chart_df.columns = wind_chart_df.columns.astype("int64")


#from dotenv import load_dotenv 

# Load the environment variables from the .env file
#load_dotenv()

# Get the value of your environment key
#env_key_value = os.getenv('OWM_KEY')

# Create a variable for your environment key
#my_var = env_key_value

st.set_page_config(
    page_title = 'Marc Leipold - Weather Forecast - Project 3', 
    page_icon=":tornado:", 
)

st.title("Kiteboarding Wind Forecast 🌧️🌥️")
st.subheader("A weather forecasting app for Kiteboarders")

city=st.text_input("ENTER THE NAME OF THE CITY ", key=None)

col1, col2 = st.columns([2,1])
with col1:
    w, wu = st.columns([3, 2])
    with w:
        weight_val = st.number_input("Enter weight", min_value=0, max_value=300, key=None)
    with wu:
        weight_unit_val = st.selectbox("Select weight unit", ["kg", "lbs"])
with col2:
    kite_size=st.selectbox("SELECT KITE SIZE ",["3m", "4m", "5m", "6m", "7m", "8m", "9m", "10m", "11m", "12m", "13m", "14m", "15m", "16m", "17m", "18m", "19m"])


col1, col2 = st.columns(2)
with col1:
    unit=st.selectbox("SELECT TEMPERATURE UNIT ",["Celsius","Fahrenheit"])
with col2:
    speed=st.selectbox("SELECT WIND SPEED UNIT ",["Knots", "Kilometers/hour", "Metre/sec", "Miles/hour"])

graph=st.radio("SELECT GRAPH TYPE ",["Bar Graph","Line Graph"])

st.markdown(
    """
    <style>
    section.main.css-k1vhr4.egzxvld5 {
        background: url("https://extrevity.com/wp-content/uploads/2021/11/background-pic.jpg");
        BACKGROUND-SIZE: COVER;
    }
    
    .block-container.css-91z34k.egzxvld4 {
        background: white;
        margin-top: 90px;
        border-radius: 16px;
        padding: 3rem !important;
    }
    
    /**** hiding the "Press Enter to Apply" notification***/
    .css-1li7dat.effi0qh1 {
    visibility: hidden;
    }
  
    </style>
    """,
    unsafe_allow_html=True
)

if weight_unit_val=="kg":
    weight_unit=" kg"
else:
    weight_unit=" lbs"
    
if unit=="Celsius":
    temp_unit=" °C"
else:
    temp_unit=" °F"
    
if speed=="Kilometers/hour":
    wind_unit=" km/h"
elif speed=="Knots":
    wind_unit=" kt"
elif speed=="Miles/hour":
    wind_unit=" mi/h"
else:
    wind_unit=" m/s"

api= "9b833c0ea6426b70902aa7a4b1da285c"
url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api}"
response=requests.get(url)
x=response.json()
    
if(st.button("SUBMIT")):
    try:
        lon=x["coord"]["lon"]
        lat=x["coord"]["lat"]
        ex="current,minutely,hourly"
        url2=f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={ex}&appid={api}'
        res=requests.get(url2)
        y=res.json()

        maxtemp=[]
        mintemp=[]
        pres=[]
        humd=[]
        wspeed=[]
        wgust=[]
        desc=[]
        cloud=[]
        rain=[]
        dates=[]
        sunrise=[]
        sunset=[]
        cel=273.15
        
      
        
        for item in y["daily"]:
            
            if unit=="Celsius":
                maxtemp.append(round(item["temp"]["max"]-cel,2))
                mintemp.append(round(item["temp"]["min"]-cel,2))
            else:
                maxtemp.append(round((((item["temp"]["max"]-cel)*1.8)+32),2))
                mintemp.append(round((((item["temp"]["min"]-cel)*1.8)+32),2))

            if wind_unit==" m/s":
                wspeed.append(str(round(item["wind_speed"]))+wind_unit)
                
            elif wind_unit==" kt":
                wspeed.append(str(round(item["wind_speed"]*1.94384)))
                wgust.append(str(round(item["wind_gust"]*1.94384)))

            elif wind_unit==" mi/h":
                wspeed.append(str(round(item["wind_speed"]*2.23694))+wind_unit)
                
            else:
                wspeed.append(str(round(item["wind_speed"]*3.6))+wind_unit)

            pres.append(item["pressure"])
            humd.append(str(item["humidity"])+' %')
            
            cloud.append(str(item["clouds"])+' %')
            rain.append(str(int(item["pop"]*100))+'%')

            desc.append(item["weather"][0]["description"].title())

            d1=datetime.date.fromtimestamp(item["dt"])
            dates.append(d1.strftime('%d %b'))
            
            sunrise.append( datetime.datetime.utcfromtimestamp(item["sunrise"]).strftime('%H:%M'))
            sunset.append( datetime.datetime.utcfromtimestamp(item["sunset"]).strftime('%H:%M'))
            
        
        
        
        def get_cell_value(weight_val, wspeed, dataframe):
            # Ensure weight and wind_speed are within the DataFrame's bounds
                if weight_val in dataframe.index and int(wspeed) in dataframe.columns:
                    return dataframe.loc[weight_val, wspeed]
                else:
                    raise ValueError("Weight and/or wind speed not found in DataFrame.")   
                
        
        def add_flag_to_bar_chart(x, y, flag_img_path, flag_img_size):
            # Create the bar chart
            fig, ax = plt.subplots()
            ax.bar(x, y, color="red")
            
            #index = np.arange(len(columns)) + 0.3
            
            # Set axis labels and title
            ax.set_xlabel("X-axis Label")
            ax.set_ylabel("Y-axis Label")
            ax.set_title("Bar Chart with Kites")
            
            # Set y-axis ticks and limits
            #ax.set_yticks(range(0, 21))  # Creates a range from 0 to 20
            #ax.set_ylim(0, 20)  # Sets the y-axis limits to 0 and 20

            # Load the flag image and create an offset image object
            flag_img = Image.open(flag_img_path)
            flag_img.thumbnail(flag_img_size)
            offset_img = OffsetImage(flag_img, zoom=1.0)
            offset_img.image.axes = ax

            # Loop through the bars and add the flag image to each one
            for i, rect in enumerate(ax.patches):
                x_pos = rect.get_x() + rect.get_width() / 2.0
                y_pos = rect.get_y() + rect.get_height() - 4.0
                ab = AnnotationBbox(offset_img, (x_pos, y_pos), xycoords='data', frameon=False)
                ax.add_artist(ab)
            

            # Return the figure
            return fig
        
        def bargraph_wind():
            fig=go.Figure(data=
                [
                go.Bar(name="Wind Speed",x=dates,y=wspeed,marker_color='green', marker=dict(line=dict(width=1, color='white'))),
                ])
            fig.update_layout(xaxis_title="Dates",yaxis_title="Wind (kts)",barmode='group',margin=dict(l=70, r=10, t=80, b=80),font=dict(color="white"), bargap=0)
            st.plotly_chart(fig)
        
        def area_chart_wind(dates, wspeed, wgust):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=wspeed, name="Wind Speed", fill='tozeroy', line_color='green'))
            fig.add_trace(go.Scatter(x=dates, y=wgust, name="Wind Gust", fill='tozeroy', line_color='red'))
            fig.update_layout(xaxis_title="Dates", yaxis_title="Wind (kts)", margin=dict(l=70, r=10, t=80, b=80), font=dict(color="white"))
            st.plotly_chart(fig)
                    
        def linegraph():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=wspeed, name='Wind Speed '))
            fig.add_trace(go.Scatter(x=dates, y=wgust, name='Wind Gust ',marker_color='crimson'))
            fig.update_layout(xaxis_title="Dates",yaxis_title="Wind (kts)",font=dict(color="white"))
            st.plotly_chart(fig)
            
        icon=x["weather"][0]["icon"]
        current_weather=x["weather"][0]["description"].title()
        
        if unit=="Celsius":
            temp=str(round(x["main"]["temp"]-cel,2))
        else:
            temp=str(round((((x["main"]["temp"]-cel)*1.8)+32),2))
                  
    
        col1, col2 = st.columns(2)
        with col1:
            st.write("## Current Temperature ")
        with col2:
            st.image(f"http://openweathermap.org/img/wn/{icon}@2x.png",width=70)


        col1, col2= st.columns(2)
        col1.metric("TEMPERATURE",temp+temp_unit)
        col2.metric("WEATHER",current_weather)
        st.subheader(" ")
        
        # st.write(y)
        

        # Define the flag image path
        flag_img_path = 'images/4Artboard 1@4x.png'

        # Define the flag image size
        flag_img_size = (50, 25)

        # Create the Streamlit app
        st.title('Kitesurfing Forecast 🌪️')
        st.write(f'This is the wind forecast for {city} for the next 8 days.')

        # Add the flag to the bar chart
        fig = add_flag_to_bar_chart(dates, wspeed, flag_img_path, flag_img_size)

        
        # Fig3 code

        def bargraph_wind3(dates, wspeed):
            # Create the bar chart
            fig3, ax = plt.subplots()
            ax.bar(dates, wspeed, color='green', edgecolor='white', linewidth=1)

            # Set axis labels and title
            ax.set_xlabel("Dates")
            ax.set_ylabel("Wind (kts)")

            # Set the y-ticks
            y_tick_interval = 2  # Choose the interval between y-ticks
            y_tick_interval = int(y_tick_interval)
            max_wspeed = max(wspeed)
            ax.set_yticks(range(0, max_wspeed + y_tick_interval, y_tick_interval))

            # Adjust layout
            ax.margins(x=0.01, y=0.01)  # Adjust margins

            # Set the spacing between bars
            bar_width = 10
            ax.set_xticks(np.arange(len(dates)))
            ax.set_xticklabels(dates)

            # Set the font color
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color("black")
            
            # Return the figure
            return fig3
        
        # Convert wspeed list elements to integers
        wspeed = [int(w) for w in wspeed]

        # Call the function with data from the DataFrame
        fig3 = bargraph_wind3(dates, wspeed)
        st.pyplot(fig3)
        
        # Show the plot in Streamlit
        st.pyplot(fig)
        

        
        
        st.write("Kite Size")
        kite_value_0 = get_cell_value(weight_val, int(wspeed[0]), wind_chart_df)
        st.write(kite_value_0)  
        
        st.write("Kite Size")
        kite_value_1 = get_cell_value(weight_val, int(wspeed[1]), wind_chart_df)
        st.write(kite_value_1)
        
        if graph=="Bar Graph":
            #area_chart_wind(dates, wspeed, wgust)
            bargraph_wind()
            
        elif graph=="Line Graph":
            linegraph()

         
        table1=go.Figure(data=[go.Table(header=dict(
                  values = [
                  '<b>DATES</b>',
                  '<b>MAX TEMP<br>(in'+temp_unit+')</b>',
                  '<b>MIN TEMP<br>(in'+temp_unit+')</b>',
                  '<b>CHANCES OF RAIN</b>',
                  '<b>CLOUD COVERAGE</b>',
                  '<b>HUMIDITY</b>'],
                  line_color='black', fill_color='royalblue',  font=dict(color='white', size=14),height=32),
        cells=dict(values=[dates,maxtemp,mintemp,rain,cloud,humd],
        line_color='black',fill_color=['paleturquoise',['palegreen', '#fdbe72']*7], font_size=14,height=32
            ))])

        table1.update_layout(margin=dict(l=10,r=10,b=10,t=10),height=328)
        st.write(table1)
        
        table2=go.Figure(data=[go.Table(columnwidth=[1,2,1,1,1,1],header=dict(values=['<b>DATES</b>','<b>WEATHER CONDITION</b>','<b>WIND SPEED</b>','<b>WIND GUST</b>','<b>PRESSURE<br>(in hPa)</b>','<b>SUNRISE<br>(in UTC)</b>','<b>SUNSET<br>(in UTC)</b>']
                  ,line_color='black', fill_color='royalblue',  font=dict(color='white', size=14),height=36),
        cells=dict(values=[dates,desc,wspeed,wgust,pres,sunrise,sunset],
        line_color='black',fill_color=['paleturquoise',['palegreen', '#fdbe72']*7], font_size=14,height=36))])
        
        table2.update_layout(margin=dict(l=10,r=10,b=10,t=10),height=360)
        st.write(table2)
        
         # just messing around - delete before deploying - ML :) 
        
       
        
        
        st.header(' ')
        st.header(' ')
        st.markdown("Made with :heart: by : ")
        st.markdown("Marc Leipold")

        

        
        
        
    except KeyError:
        st.error(" Invalid city!!  Please try again !!")


       