import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def initiate_vectroizer():
    max_vocab_length = 10000
    average_length = 8
    max_length = average_length #8
    #max_length = 8
    text_Vectorizer = tf.keras.layers.TextVectorization (max_tokens= max_vocab_length,  # how many words in vocab
                                                        #  standardize="lower_and_strip_punctuation",
                                                        #  split= "whitespace",
                                                        #  ngrams= None,
                                                        output_mode="int", # can be function but we need int
                                                        output_sequence_length= max_length # We dont cap the output as longest sequence is unknown it will basically pad others
                                                        #  pad_to_max_tokens=True
    )
    # text_Vectorizer.adapt(X_train)
    return text_Vectorizer

def initiate_embedder():
    embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length,# Input shape or Vocab
                        output_dim=128,
                        input_length=max_length# Length of Sequence
                        )
    return embedding

def encode_labels(y_train,y_valid):

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_valid)

    return y_train_encoded,y_val_encoded
