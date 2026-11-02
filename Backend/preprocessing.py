import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Constants and Paths to your saved artifacts
MODEL_PATH = "Transformer_Model.keras"
TOKENIZER_PATH = "TFtokenizer.pkl"
IMG_SIZE = 224
MAX_SEQ_LENGTH = 40


def load_artifacts():
    """Loads the Transformer model and tokenizer into RAM once when the server starts."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found at: {TOKENIZER_PATH}")

    print("[INFO] Loading Tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    start_token_id = tokenizer.word_index["<start>"]

    print("[INFO] Loading Transformer Model...")
    # compile=False skips loading Adam optimizer & custom masked_accuracy metric for faster inference
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Warm-up pass: compiles the TensorFlow graph on dummy data so the 1st camera frame is fast
    dummy_img = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    dummy_seq = tf.constant([[start_token_id]], dtype=tf.int32)
    _ = model([dummy_img, dummy_seq], training=False)

    print("[SUCCESS] Model and Tokenizer loaded and warmed up!")
    return model, tokenizer


def preprocess_image_bytes(raw_bytes: bytes):
    """Decodes raw JPEG bytes from the API request and formats pixels for EfficientNetB0."""
    # NOTE: Do NOT use tf.io.read_file() here because raw_bytes is already in memory!
    img = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])  # decode_image doesn't set static shape; resize needs it
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = keras.applications.efficientnet.preprocess_input(img)
    return img


def decode_tokenizer_sequence(token_ids, tokenizer_obj):
    """Converts a list of integer token IDs back into a readable English sentence."""
    words = []
    for idx in token_ids:
        if idx == 0:
            continue
        word = tokenizer_obj.index_word.get(idx, "<unk>")
        if word == "<start>":
            continue
        if word == "<end>":
            break
        words.append(word)
    return " ".join(words)


def generate_caption(model, img_tensor, tokenizer_obj, max_len=MAX_SEQ_LENGTH):
    """Generates 1 caption for an image by picking the highest-scoring word at each step."""
    start_token_id = tokenizer_obj.word_index["<start>"]
    end_token_id = tokenizer_obj.word_index["<end>"]

    img_batch = tf.expand_dims(img_tensor, axis=0)
    caption_seq = [start_token_id]

    for _ in range(max_len):
        seq_input = tf.constant([caption_seq], dtype=tf.int32)
        logits = model([img_batch, seq_input], training=False)
        last_word_logits = logits[0, -1, :].numpy()

        # Block <PAD> (0) and <unk> so the model only picks real words
        last_word_logits[0] = -1e9
        if "<unk>" in tokenizer_obj.word_index:
            last_word_logits[tokenizer_obj.word_index["<unk>"]] = -1e9

        next_word_id = int(np.argmax(last_word_logits))
        caption_seq.append(next_word_id)

        if next_word_id == end_token_id:
            break

    return decode_tokenizer_sequence(caption_seq, tokenizer_obj)