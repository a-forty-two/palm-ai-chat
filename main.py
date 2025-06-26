from flask import Flask, render_template, request, session
import os
import yaml
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, Part, HarmCategory, HarmBlockThreshold, Content # Import Content

app = Flask(__name__)
app.secret_key = "session_key1"

def get_config_value(config, section, key, default=None):
    """
    Retrieve a configuration value from a section with an optional default value.
    """
    try:
        return config[section][key]
    except:
        return default
        
with open('config.yaml') as f:
    config = yaml.safe_load(f)

TITLE = get_config_value(config, 'app', 'title', 'Ask Google')
SUBTITLE = get_config_value(config, 'app', 'subtitle', 'Your friendly Bot')
CONTEXT = get_config_value(config, 'gemini', 'context', 'You are a bot who can answer all sorts of questions')
BOTNAME = get_config_value(config, 'gemini', 'botname', 'Google')
TEMPERATURE = get_config_value(config, 'gemini', 'temperature', 0.8)
MAX_OUTPUT_TOKENS = get_config_value(config, 'gemini', 'max_output_tokens', 256)
TOP_P = get_config_value(config, 'gemini', 'top_p', 0.8)
TOP_K = get_config_value(config, 'gemini', 'top_k', 40)

EXAMPLES = get_config_value(config, 'gemini', 'examples', [])
gemini_examples_content = [] # Changed name to reflect it holds Content objects
for ex_input, ex_output in EXAMPLES.items():
    # Each example pair needs to be two Content objects: one for user, one for model
    gemini_examples_content.append(Content(role="user", parts=[Part.from_text(ex_input)]))
    gemini_examples_content.append(Content(role="model", parts=[Part.from_text(ex_output)]))
      
@app.route("/", methods = ['POST', 'GET'])
def main():
    if request.method == 'POST':
        input = request.form['input']
        if input and input.strip(): # Check if input is not just whitespace
            response = get_response(input)
        else:
            response = "Ask me something"
        input = ""
    else:
        input = ""
        session.pop('chat_history', None)       
        response = get_response("Who are you and what can you do?")


    model_data = {"title": TITLE, "subtitle": SUBTITLE, "botname": BOTNAME, "message": response, "input": input, "history": session.get("chat_history", [])}
    return render_template('index.html', model=model_data)


def get_response(input):
    vertexai.init(location="us-central1")
    
    # System instruction for Gemini (replaces context) - Pass to GenerativeModel constructor
    system_instruction_part = Part.from_text(CONTEXT) if CONTEXT else None

    # Use GenerativeModel for Gemini, passing system_instruction here
    # THIS IS THE CRITICAL CHANGE: Use "gemini-2.5-flash" directly
    model = GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction_part) 

    # Safety settings for Gemini
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }

    generation_config = {
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "top_p": TOP_P,
        "top_k": TOP_K
    }

    chat_history_for_gemini_content = [] # This will hold the Content objects for the chat session
    if 'chat_history' in session:
        # Convert existing chat history from session to Gemini's expected Content object format
        for item in session["chat_history"]:
            chat_history_for_gemini_content.append(Content(role=item["author"], parts=[Part.from_text(item["content"])]))

    # Prepend few-shot examples (as Content objects) to the history
    full_chat_history = gemini_examples_content + chat_history_for_gemini_content

    # Start chat with the full conversation history (now a list of Content objects)
    chat = model.start_chat(history=full_chat_history)

    # Send message directly. The ChatSession automatically wraps it in a Content(role="user", ...)
    response = chat.send_message(
        input, # Pass the input string directly
        generation_config=generation_config, 
        safety_settings=safety_settings
    )
    
    # Update session chat history (still storing simple dicts for rendering)
    session_history = session.get("chat_history", [])
    session_history.append({"content": input, "author": "user"})
    session_history.append({"content": response.text, "author": "model"})
    session["chat_history"] = session_history
    
    return response.text
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
