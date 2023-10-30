import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


data = [
    {"en": "What kind of movie is it", "hi_ng": "yah kis tarah kee philm hai"},
    {"en": "When was the movie made?", "hi_ng": "film kab banee thee?"}
    # Add more data here
]


with open('/Users/arshitarora/projects/English-Hinglish/test.json', 'r') as file:
    data = json.load(file)


english_texts = [entry['en'] for entry in data]
hinglish_texts = [entry['hi_ng'] for entry in data]



en_tokenizer = Tokenizer(filters='')
en_tokenizer.fit_on_texts(english_texts)
en_seq = en_tokenizer.texts_to_sequences(english_texts)

# Tokenizing Hinglish text
hi_tokenizer = Tokenizer(filters='')
hi_tokenizer.fit_on_texts(hinglish_texts)
hi_seq = hi_tokenizer.texts_to_sequences(hinglish_texts)


en_seq = pad_sequences(en_seq, padding='post')
hi_seq = pad_sequences(hi_seq, padding='post')


latent_dim = 256  # Dimensionality of the encoding space

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = tf.keras.layers.Embedding(len(en_tokenizer.word_index) + 1, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = tf.keras.layers.Embedding(len(hi_tokenizer.word_index) + 1, latent_dim, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(len(hi_tokenizer.word_index) + 1, activation='softmax')
output = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')


model.fit([en_seq, hi_seq[:,:-1]], hi_seq[:,1:], batch_size=64, epochs=250, validation_split=0.2)


encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Functions for translation
def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = 0  
    stop_condition = False
    translated_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in hi_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        if sampled_word == 'end' or len(translated_sentence.split()) > 50:
            stop_condition = True
        else:
            if sampled_word is not None:
                translated_sentence += sampled_word + ' '
                target_seq[0, 0] = sampled_token_index
            else:
                stop_condition = True  
            states_value = [h, c]
    return translated_sentence

# Translate an English sentence
user_input = input("Enter an English sentence: ")
input_sequence = en_tokenizer.texts_to_sequences([user_input])
input_sequence = pad_sequences(input_sequence, maxlen=en_seq.shape[1], padding='post')
translated = translate_sentence(input_sequence)
print("Translated to Hinglish:", translated)
