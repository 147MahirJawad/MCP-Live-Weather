from fastapi import FastAPI, HTTPException
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import json
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Define the tool for Gemini
get_weather_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for a specified location.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "location": genai.protos.Schema(type=genai.protos.Type.STRING, description="The city name to get weather for"),
                },
                required=["location"],
            ),
        )
    ]
)

# Initialize model with tools
model = genai.GenerativeModel('gemini-2.5-flash', tools=[get_weather_tool])
API_KEY = os.getenv("API_KEY") # This is your OpenWeatherMap API key

# Weather API function
def get_weather(location: str) -> dict:
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Error: Weather API key not configured")
        return {"error": "Weather API key not configured"}

    try:
        response = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"q": location, "appid": api_key, "units": "metric"}, # Added units: metric
            timeout=10
        )
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Error fetching weather: Request to OpenWeatherMap timed out for {location}")
        return {"error": "Request to weather API timed out."}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather for {location}: {e}")
        return {"error": f"Error connecting to weather API: {str(e)}"}
    except Exception as e:
        print(f"Unexpected error in get_weather for {location}: {e}")
        return {"error": str(e)}

# Tool execution function
def execute_tool(tool_name: str, parameters: dict) -> str:
    if tool_name == "get_weather":
        # The 'London' fallback will now only be used if Gemini genuinely doesn't provide a location
        result = get_weather(parameters.get("location", "London"))
        if "error" in result:
            return result["error"]

        try:
            # Correct parsing for OpenWeatherMap API
            location_name = result.get("name", "Unknown City")
            country = result.get("sys", {}).get("country", "Unknown Country")
            temp_c = result.get("main", {}).get("temp")
            humidity = result.get("main", {}).get("humidity")
            wind_speed_ms = result.get("wind", {}).get("speed")
            weather_condition_list = result.get("weather", [])
            weather_condition = weather_condition_list[0].get("description", "N/A") if weather_condition_list else "N/A"
            feels_like_c = result.get("main", {}).get("feels_like")

            if None in [temp_c, humidity, wind_speed_ms, feels_like_c]:
                return "Incomplete weather data received from API."

            # Convert wind speed from m/s to km/h
            wind_kph = wind_speed_ms * 3.6

            return (
                f"Weather in {location_name}, {country}:\n"
                f"• Temperature: {temp_c:.1f}°C\n"
                f"• Condition: {weather_condition.capitalize()}\n"
                f"• Humidity: {humidity}%\n"
                f"• Wind: {wind_kph:.1f} km/h\n"
                f"• Feels like: {feels_like_c:.1f}°C"
            )
        except KeyError as e:
            print(f"Error parsing weather data: Missing key {e}. Response: {result}")
            return f"Error parsing weather data: Missing key {e}. Please check the API response structure."
        except Exception as e:
            print(f"An unexpected error occurred during weather parsing: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"
    else:
        return f"Unknown tool: {tool_name}"

# AI processing with tool selection (using Function Calling)
def process_with_tools(query: str) -> str:
    try:
        response = model.generate_content(query)

        # Check if Gemini wants to call a function
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_name = part.function_call.name
                    
                    function_args = {}
                    if hasattr(part.function_call, 'args'):
                        # Directly convert MapComposite or dict-like objects to dict
                        if hasattr(part.function_call.args, 'items'):
                            function_args = dict(part.function_call.args)
                        elif isinstance(part.function_call.args, str):
                            try:
                                parsed_args = json.loads(part.function_call.args)
                                if isinstance(parsed_args, dict):
                                    function_args = parsed_args
                                else:
                                    print(f"Warning: function_call.args was string but not a dict after json.loads: {parsed_args}")
                            except json.JSONDecodeError:
                                print(f"Warning: function_call.args was a string but not valid JSON: {part.function_call.args}")
                        else:
                            print(f"Warning: Unexpected type for function_call.args: {type(part.function_call.args)}")


                    if function_name == "get_weather":
                        print(f"Gemini requested tool: {function_name} with args: {function_args}")
                        tool_result = execute_tool("get_weather", function_args)
                        print(f"Tool execution result: {tool_result}")

                        # Send the tool result back to Gemini for a final, natural language response
                        chat_session = model.start_chat()
                        chat_session.send_message(query) # Send user's original query to establish context
                        
                        response_with_tool_output = chat_session.send_message(
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name="get_weather",
                                    response={"result": tool_result} # Wrap the string result in a dict for FunctionResponse
                                )
                            )
                        )
                        return response_with_tool_output.text
                    else:
                        return f"Unknown tool requested by AI: {function_name}"
                elif part.text:
                    # Gemini provided a direct text response (no tool needed or tool not applicable)
                    return part.text
        return "No coherent response from AI (neither text nor function call)."

    except Exception as e:
        print(f"Error during AI processing in process_with_tools: {e}")
        return f"AI Error: {str(e)}"

# FastAPI endpoints
@app.post("/api/query")
async def handle_query(payload: dict):
    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    response = process_with_tools(query)
    return {"response": response}

# Gradio UI
def gradio_interface(query: str):
    return process_with_tools(query)

gradio_app = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Your Message"),
    outputs=gr.Textbox(label="MCP Response"),
    title="MCP Server",
    description="Multi-Component Processing Server with Gemini AI and Weather Tools"
)

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

# Health check
@app.get("/")
def health_check():
    return {"status": "active", "components": ["fastapi", "gemini", "weather-api", "gradio"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)