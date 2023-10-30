import openai
import gradio as gr

# Set your OpenAI API key here
api_key = ""
openai.api_key = api_key

def english_to_hinglish(english_sentence):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Translate the following English sentence to Hinglish: '{english_sentence}'",
        max_tokens=100,
        temperature=0.0,
        n=1,
        stop=None
    )

    hinglish_translation = response.choices[0].text.strip()
    return hinglish_translation

#uncomment this to use the transaltor in terminal

# english_text = input("Enter the English text to translate to Hinglish: ")
# hinglish_text = english_to_hinglish(english_text)
# print(f"English: {english_text}")
# print(f"Hinglish: {hinglish_text}")

#gradio interface
iface = gr.Interface(
    fn=english_to_hinglish,
    inputs="text",
    outputs="text",
    title="English to Hinglish Translator",
    description="Translate English text to Hinglish using OpenAI's GPT-3 model."
)

iface.launch()