import requests
import streamlit as st


def get_groq_response(input_text):
    json_body = {
        "input": {
            "language": "Tamil",
            "text": input_text
        }
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/chain/invoke", json=json_body)
        # print(response.json()['output'])
        return response.json()
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        return None


# Streamlit app
st.title("LLM Application Using LCEL")
input_text = st.text_input("Enter the text you want to convert to french")

if input_text:
    st.write(
        f"{input_text} in French is {get_groq_response(input_text)['output']}")
