import streamlit as st
import google.generativeai as genai
import pyttsx3
import speech_recognition as sr
import requests
import json
import threading
import pygame
import cv2
import subprocess
from datetime import datetime

# Initialize pygame for playing audio
pygame.mixer.init()

# Configure the Gemini model
genai.configure(api_key="YOUR_API_KEY_HERE")

instruction2llm = '''
"Please respond only in JSON format. When I provide text for analysis, respond with the appropriate module, its parameters, the count of those parameters, a success flag, and a brief description. If you cannot understand the input, respond with 'Sorry, I can't understand.' Here are additional modules to be integrated into our talking robot program, along with examples of inputs and their expected outputs, Send to GeminiModule if you cant understand :

Modules:
1. 'Weather' - Get weather updates with 'City'.
2. 'Reminder' - Set reminders with 'Time', 'Message'.
3. 'Translator' - Translate text with 'Source Language', 'Target Language'.
4. 'News' - Fetch news with 'Category', 'Region'.
5. 'Jokes' - Tell jokes with 'Category', 'Language'.
6. 'CurrencyConverter' - Convert currency with 'From Currency', 'To Currency'.
7. 'SportsScores' - Provide sports scores with 'Sport Type', 'League'.
8. 'MusicPlayer' - Play music with 'Song Title', 'Artist'.
9. 'Navigation' - Provide directions with 'Start Location', 'End Location'.
10. 'Timer' - Set a timer with 'Duration'.
11. 'CookingRecipes' - Suggest recipes with 'Dish Name', 'Cuisine'.
12. 'FitnessTracker' - Track fitness activities with 'Activity Type', 'Duration'.
13. 'Quiz' - Conduct quizzes with 'Topic', 'Difficulty Level'.
14. 'HealthAdvice' - Give health advice with 'Symptom', 'Age Group'.
15. 'AppointmentScheduler' - Schedule appointments with 'Date', 'Time', 'Purpose'.
16. 'LearningAssistant' - Provide educational content with 'Subject', 'Grade Level'.
17. 'VoiceControl' - Control devices with 'Device Name', 'Action'.
18. 'EmergencyAlert' - Send emergency alerts with 'Type of Emergency', 'Location'.
19. 'TravelPlanner' - Plan trips with 'Destination', 'Departure Date', 'Budget'.
20. 'PetCare' - Provide pet care advice with 'Pet Type', 'Issue'.
21. 'EventReminder' - Remind about events with 'Event Name', 'Event Date'.
22. 'LanguageLearning' - Assist in language learning with 'Language', 'Proficiency Level'.
23. 'MoodTracker' - Track mood with 'Current Mood', 'Time of Day'.
24. 'ProductFinder' - Help find products with 'Product Name', 'Category'.
25. 'Horoscope' - Provide horoscope with 'Zodiac Sign'.
26. 'BookRecommendations' - Suggest books with 'Genre', 'Author'.
27. 'MovieSuggestions' - Recommend movies with 'Genre', 'Age Group'.
28. 'HomeworkHelper' - Assist with homework with 'Subject', 'Complexity Level'.
29. 'GardeningTips' - Offer gardening advice with 'Plant Type', 'Season'.
30. 'ArtificialIntelligence' - Explain AI concepts with 'Concept', 'Difficulty Level'.
31. 'PublicTransportInfo' - Provide transport information with 'Transport Mode', 'Route'.
32. 'StockMarketUpdates' - Give stock updates with 'Stock Symbol', 'Market'.
33. 'PersonalFinance' - Offer finance advice with 'Finance Topic', 'Experience Level'.
34. 'Time' - Provide the current time.
35. 'TakePicture' - Take a picture with the webcam.
36. 'VoiceControl' - Control devices with 'Device Name', 'Action'.
36. 'GeminiModule' - Sends to gemini to get answers from GEMINI AI

Examples:
1. Input: "What is the weather in New York?"
   Output:
   json
   {
     "module": "Weather",
     "parameters": ["New York"],
     "parameter_count": 1,
     "success": true,
     "description": "Providing weather updates for New York."
   }

2. Input: "Set a reminder for a meeting at 3 PM."
   Output:
   json
   {
     "module": "Reminder",
     "parameters": ["3 PM", "meeting"],
     "parameter_count": 2,
     "success": true,
     "description": "Reminder set for a meeting at 3 PM."
   }

3. Input: "Translate 'Hello' from English to Spanish."
   Output:
   json
   {
     "module": "Translator",
     "parameters": ["English", "Spanish", "Hello"],
     "parameter_count": 3,
     "success": true,
     "description": "Translating 'Hello' from English to Spanish."
   }

4. Input: "Tell me a joke about animals."
   Output:
   json
   {
     "module": "Jokes",
     "parameters": ["animals"],
     "parameter_count": 1,
     "success": true,
     "description": "Telling a joke about animals."
   }

5. Input: "How do I invest in Apple stocks?"
   Output:
   json
   {
     "module": "StockMarketUpdates",
     "parameters": ["Apple"],
     "parameter_count": 1,
     "success": true,
     "description": "Providing investment information for Apple stocks."
   }

6. Input: "What is the current time?"
   Output:
   json
   {
     "module": "Time",
     "parameters": [],
     "parameter_count": 0,
     "success": true,
     "description": "Providing the current time."
   }

7. Input: "Open Notepad"
   Output:
   json
   {
     "module": "VoiceControl",
     "parameters": ["Notepad", "open"],
     "parameter_count": 2,
     "success": true,
     "description": "Opening Notepad."
   }

8. Input: "I don't know what to do."
   Output:
   json
   {
     "response": "Sorry, I can't understand.",
     "success": false,
     "description": "The input could not be understood."
   }
9. Input: "How Wifi Works"
   Output:
   json
   {
     "response": "GeminiModule",
     "parameters": ["How Wifi Works"],
     "parameter_count": 1,
     "success": true,
     "description": "General Question on How Wifi Works"
   }


'''

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to recognize speech input
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.write("Please say something...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            st.write("Timeout. No speech detected.")
            return ""

    try:
        st.write("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        st.write(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        st.write("Sorry, the speech recognition service is unavailable.")
        return ""

# Function to take a picture
def take_picture():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"error": "Could not open webcam."}

        ret, frame = cap.read()
        if not ret:
            return {"error": "Failed to capture image."}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"picture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        cap.release()
        return {"success": True, "filename": filename, "description": f"Picture saved as {filename}"}
    except Exception as e:
        return {"error": str(e)}

# Function to handle API requests based on the module
def handle_api_request(json_response):
    module = json_response.get("module")
    parameters = json_response.get("parameters")

    if module == "Weather":
        city = parameters[0]
        api_key = "YOUR_WEATHER_API_KEY_HERE"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return {"error": response.get("message", "Error fetching weather data.")}
        
        weather = response["weather"][0]["description"]
        temperature = response["main"]["temp"]
        humidity = response["main"]["humidity"]
        wind_speed = response["wind"]["speed"]
        
        weather_info = (f"Current weather in {city}: {weather}. "
                        f"Temperature: {temperature}°C, "
                        f"Humidity: {humidity}%, "
                        f"Wind Speed: {wind_speed} m/s.")
        
        return {"weather_info": weather_info}

    elif module == "CurrencyConverter":
        return "Implement currency converter API functionality here"

    elif module == "StockMarketUpdates":
        stock_symbol = parameters[0]
        api_key = "YOUR_STOCK_API_KEY_HERE"
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock_symbol}&interval=5min&apikey={api_key}"
        response = requests.get(url).json()

        if "Time Series (5min)" not in response:
            return {"error": "Error fetching stock data."}

        time_series = response["Time Series (5min)"]
        latest_timestamp = next(iter(time_series))
        latest_data = time_series[latest_timestamp]

        stock_info = (f"Stock update for {stock_symbol}: "
                      f"Price: {latest_data['1. open']}, "
                      f"High: {latest_data['2. high']}, "
                      f"Low: {latest_data['3. low']}, "
                      f"Close: {latest_data['4. close']}, "
                      f"Volume: {latest_data['5. volume']}.")
        return {"stock_info": stock_info}

    elif module == "Jokes":
        url = "https://v2.jokeapi.dev/joke/Any"
        response = requests.get(url).json()
        if response.get("error"):
            return {"error": "Error fetching joke."}

        if response["type"] == "single":
            joke = response["joke"]
        else:
            joke = f"{response['setup']} ... {response['delivery']}"

        return {"joke": joke}

    elif module == "GeminiModule":
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": [""]},
            ]
        )
        response = chat_session.send_message(f"Limit to 100 words.\n {parameters[0]}")
        return {"GeminiModule": response.text}

    elif module == "Time":
        current_time = datetime.now().strftime("%H:%M:%S")
        return {"time": current_time}

    elif module == "Timer":
        duration = parameters[0]
        try:
            duration_parts = duration.split()
            value = int(duration_parts[0])
            unit = duration_parts[1].lower() if len(duration_parts) > 1 else "seconds"
            if unit.startswith("minute"):
                seconds = value * 60
            elif unit.startswith("second"):
                seconds = value
            else:
                return {"error": "Invalid duration unit."}

            threading.Timer(seconds, play_time_up_message).start()
            return {"timer": seconds}
        except (IndexError, ValueError):
            return {"error": "Invalid duration format."}

    elif module == "TakePicture":
        engine.say("Sure, I'll take a picture now, Keep Smile.")
        engine.runAndWait()
        response = take_picture()
        if response.get("success"):
            engine.say(response["description"])
            engine.say(f"Picture saved as {response['filename']}.")
            engine.runAndWait()
        return response

    elif module == "VoiceControl":
        device_name = parameters[0]
        action = parameters[1]
        if action.lower() == "open":
            if device_name.lower() == "calculator":
                subprocess.Popen("calc.exe")
                return {"description": "Opening Calculator."}
            elif device_name.lower() == "chrome":
                subprocess.Popen("chrome.exe")
                return {"description": "Opening Browser."}
            elif device_name.lower() == "notepad":
                subprocess.Popen("notepad.exe")
                return {"description": "Opening Notepad."}

        elif action.lower() == "close":
            if device_name.lower() == "calculator":
                subprocess.Popen("taskkill /IM calc.exe /F")
                return {"description": "Closing Calculator"}
            elif device_name.lower() == "chrome":
                subprocess.Popen("taskkill /IM chrome.exe /F")
                return {"description": "Closing Chrome"}
            elif device_name.lower() == "notepad":
                subprocess.Popen("taskkill /IM notepad.exe /F")
                return {"description": "Closing Notepad."}

    return {"error": "Sorry, I can't understand."}

def play_time_up_message():
    try:
        pygame.mixer.music.load("your_time_up_message_path.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        st.write("Error:", e)

def main():
    st.title("vinay BOT")

    # Text Input
    user_input = st.text_input("Enter your query:")
    
    if st.button("Submit Query"):
        json_response = model.generate(
            prompt=user_input,
            instruction=instruction2llm
        )
        
        response = handle_api_request(json.loads(json_response))
        st.json(response)
    
    # Voice Input
    if st.button("Use Voice Command"):
        voice_input = recognize_speech()
        if voice_input:
            json_response = model.generate(
                prompt=voice_input,
                instruction=instruction2llm
            )
            response = handle_api_request(json.loads(json_response))
            st.json(response)

if __name__ == "__main__":
    main()
