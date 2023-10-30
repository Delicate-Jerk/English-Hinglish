import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('eng_to_hinglish_model.h5')

# Define the required parameters (based on your previous training)
max_encoder_seq_length = 10# Define the maximum encoder sequence length
num_encoder_tokens = 10# Define the number of tokens for the encoder

# These dictionaries were used during training and should be available with the same values used in training
input_token_index = {}  # Populate with the values used during training
target_token_index = {}  # Populate with the values used during training
reverse_target_char_index = {}  # Populate with the values used during training

# Define a function to preprocess input sentences for translation
def preprocess_input_sentence(input_sentence):
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')

    for t, char in enumerate(input_sentence):
        encoder_input_data[0, t, input_token_index[char]] = 1.0

    return encoder_input_data

# Define a function to perform the translation
def translate(input_sentence):
    encoder_input_data = preprocess_input_sentence(input_sentence)
    decoded_sentence = ''

    # Define the decoder input (initially only '\t')
    decoder_input = np.zeros((1, 1, len(target_token_index)))
    decoder_input[0, 0, target_token_index['\t']] = 1.0

    while True:
        output_tokens = model.predict([encoder_input_data, decoder_input])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            break

        decoded_sentence += sampled_char

        # Update the decoder input for the next time step
        decoder_input = np.zeros((1, 1, len(target_token_index)))
        decoder_input[0, 0, sampled_token_index] = 1.0

    return decoded_sentence

# Usage example
input_sentence = "Your English sentence to be translated"
translated_sentence = translate(input_sentence)
print("Translated Hinglish sentence:", translated_sentence)
