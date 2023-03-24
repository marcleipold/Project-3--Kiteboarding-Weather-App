import streamlit as st
import datetime,requests
from plotly import graph_objects as go
from PIL import Image
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

st.title("8-DAY WEATHER FORECAST 🌧️🌥️")

city=st.text_input("ENTER THE NAME OF THE CITY ", key=None)

col1, col2 = st.columns([2,1])
with col1:
    weight, weight_unit = st.columns([3, 2])
    with weight:
        weight_val = st.number_input("Enter weight", min_value=0, max_value=300, key=None)
    with weight_unit:
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
        desc=[]
        cloud=[]
        rain=[]
        dates=[]
        sunrise=[]
        sunset=[]
        cel=273.15
        
        # just messing around - delete before deploying - ML :) 
        
        # st.write(y)
        
        
        for item in y["daily"]:
            
            if unit=="Celsius":
                maxtemp.append(round(item["temp"]["max"]-cel,2))
                mintemp.append(round(item["temp"]["min"]-cel,2))
            else:
                maxtemp.append(round((((item["temp"]["max"]-cel)*1.8)+32),2))
                mintemp.append(round((((item["temp"]["min"]-cel)*1.8)+32),2))

            if wind_unit=="m/s":
                wspeed.append(str(round(item["wind_speed"],1))+wind_unit)
                
            elif wind_unit=="kt":
                wspeed.append(str(round(item["wind_speed"]*1.94384,1))+wind_unit)
            
            elif wind_unit=="mi/h":
                wspeed.append(str(round(item["wind_speed"]*2.23694,1))+wind_unit)
                
            else:
                wspeed.append(str(round(item["wind_speed"]*3.6,1))+wind_unit)

            pres.append(item["pressure"])
            humd.append(str(item["humidity"])+' %')
            
            cloud.append(str(item["clouds"])+' %')
            rain.append(str(int(item["pop"]*100))+'%')

            desc.append(item["weather"][0]["description"].title())

            d1=datetime.date.fromtimestamp(item["dt"])
            dates.append(d1.strftime('%d %b'))
            
            sunrise.append( datetime.datetime.utcfromtimestamp(item["sunrise"]).strftime('%H:%M'))
            sunset.append( datetime.datetime.utcfromtimestamp(item["sunset"]).strftime('%H:%M'))
            
            

        def bargraph():
            fig=go.Figure(data=
                [
                go.Bar(name="Maximum",x=dates,y=maxtemp,marker_color='crimson'),
                go.Bar(name="Minimum",x=dates,y=mintemp,marker_color='navy')
                ])
            fig.update_layout(xaxis_title="Dates",yaxis_title="Temperature",barmode='group',margin=dict(l=70, r=10, t=80, b=80),font=dict(color="white"))
            st.plotly_chart(fig)
        
        def linegraph():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=mintemp, name='Minimum '))
            fig.add_trace(go.Scatter(x=dates, y=maxtemp, name='Maximimum ',marker_color='crimson'))
            fig.update_layout(xaxis_title="Dates",yaxis_title="Temperature",font=dict(color="white"))
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
        
        if graph=="Bar Graph":
            bargraph()
            
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
        
        table2=go.Figure(data=[go.Table(columnwidth=[1,2,1,1,1,1],header=dict(values=['<b>DATES</b>','<b>WEATHER CONDITION</b>','<b>WIND SPEED</b>','<b>PRESSURE<br>(in hPa)</b>','<b>SUNRISE<br>(in UTC)</b>','<b>SUNSET<br>(in UTC)</b>']
                  ,line_color='black', fill_color='royalblue',  font=dict(color='white', size=14),height=36),
        cells=dict(values=[dates,desc,wspeed,pres,sunrise,sunset],
        line_color='black',fill_color=['paleturquoise',['palegreen', '#fdbe72']*7], font_size=14,height=36))])
        
        table2.update_layout(margin=dict(l=10,r=10,b=10,t=10),height=360)
        st.write(table2)
        
        st.header(' ')
        st.header(' ')
        st.markdown("Made with :heart: by : ")
        st.markdown("Marc Leipold")
 
    except KeyError:
        st.error(" Invalid city!!  Please try again !!")

