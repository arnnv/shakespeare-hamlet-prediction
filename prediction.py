from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model('model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(text, model=model, tokenizer=tokenizer):
    max_sequence_len = model.input_shape[1]+1
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1)

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None
