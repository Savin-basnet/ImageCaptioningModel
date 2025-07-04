# Add this at the beginning of app.py
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Blueprint 
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import io
import base64
import os
import re
import numpy as np
import matplotlib.pyplot as plt

import mysql.connector
from mysql.connector import Error
import os
from flask import current_app, session



import tensorflow as tf
from tensorflow import keras
from tensorflow import keras

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, TextVectorization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet

# blueprint/ImageCaption.py

import os
import json
import numpy as np
import tensorflow as tf
from flask import Blueprint, render_template, request, redirect, current_app, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
from tensorflow.keras import layers
import io
import base64
from PIL import Image, ImageDraw, ImageFont
from configparser import ConfigParser


# === Blueprint ===
bp_imageCaption = Blueprint('image_caption', __name__)

# databse connection 
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='swarupdb'
        )
        return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None


# config = configparser.ConfigParser()
# config.read('your_config.ini')

# some_value = config.get('section_name', 'option_name')
# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# bp_imageCaption.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# bp_imageCaption.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Model parameters (must match training)
IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512

# Initialize model components
cnn_model = None
encoder_wrapper = None
decoder_wrapper = None
vectorizer = None
encoding='utf-8'

import matplotlib.font_manager as fm

# Add this to your initialization code
font_path = "Lohit-Devanagari.ttf"
if os.path.exists(font_path):
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

# Custom layer definitions (must match training exactly)
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )

    def get_config(self):
        return {
            "post_warmup_learning_rate": self.post_warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

        

    def call(self, inputs, training, mask=None):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM,
            sequence_length=SEQ_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, caption_input, encoder_output, mask=None, training=False):
        # Fixed: Using caption_input instead of undefined 'inputs'
        inputs = self.embedding(caption_input)  # Changed from inputs to caption_input
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else:
            padding_mask = None
            combined_mask = causal_mask

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # Fixed: Using encoder_output instead of undefined 'encoder_outputs'
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_output,  # Changed from encoder_outputs to encoder_output
            key=encoder_output,     # Changed from encoder_outputs to encoder_output
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.constant([1, 1], dtype=tf.int32),
            ],
            axis=0,
        )
        return tf.tile(mask, mult)


# Alternative approach without matplotlib warnings
from PIL import Image, ImageDraw, ImageFont

def generate_caption_image(image_path, caption):
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Try to use Nepali font
        try:
            font = ImageFont.truetype("ModelFiles/Lohit-Devanagari.ttf", 20)
            print("Using custom font")
        except:
            font = ImageFont.load_default()
            print("Falling back to default font")

        # Add black semi-transparent background for text
        text_bg = Image.new('RGBA', img.size, (0,0,0,0))
        draw_bg = ImageDraw.Draw(text_bg)
        draw_bg.rectangle([10, 10, img.width-10, 60], fill=(0,0,0,128))
        
        # Combine images
        img = Image.alpha_composite(img.convert('RGBA'), text_bg)
        draw = ImageDraw.Draw(img)
        
        # Add text
        draw.text((20, 15), caption, font=font, fill="white")
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
        
    except Exception as e:
        print(f"Error creating caption image: {e}")
        return None


class EncoderModel(keras.Model):
    def __init__(self, encoder_block, **kwargs):
        super().__init__(**kwargs)
        self.encoder_block = encoder_block

    def build(self,input_shape):
        self.encoder_block.build(input_shape)
        self.build = True
        
    def call(self, inputs, training=False):
        return self.encoder_block(inputs, training=training)

class DecoderModel(keras.Model):
    def __init__(self, decoder_block, **kwargs):
        super().__init__(**kwargs)
        self.decoder_block = decoder_block

    def build(self,input_shape):
        if isinstance(input_shape,list) and len(input_shape) ==3:
            self.decoder_block.build(input_shape[0])
        self.build = True
        
    def call(self, inputs, training=False):
        # Unpack the inputs if they come as a tuple/list
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            caption_input, encoder_output, mask = inputs
            return self.decoder_block(
                caption_input, 
                encoder_output, 
                mask=mask, 
                training=training
            )
        else:
            # Handle other input formats if needed
            return self.decoder_block(inputs, training=training)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():

    # At the VERY START of load_models():
    tf.keras.backend.clear_session()
    print("Cleared TensorFlow session")

    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0],True)

        except:
            pass

    global cnn_model, encoder_wrapper, decoder_wrapper, vectorizer
    
    # Load custom objects
    custom_objects = {
        'LRSchedule': LRSchedule,
        'TransformerEncoderBlock': TransformerEncoderBlock,
        'TransformerDecoderBlock': TransformerDecoderBlock,
        'EncoderModel': EncoderModel,
        'DecoderModel': DecoderModel
    }
    
    # Load CNN model
    cnn_model = load_model('ModelFiles/cnn_model.keras', custom_objects=custom_objects)
    # After loading cnn_model:
    cnn_model.trainable = False  # Should already be frozen
    print("CNN output shape:", cnn_model.output_shape)  # Should be (None, 49, 1280)
    
    # Load encoder
    # Replace the encoder/decoder loading code with:
    encoder_block = TransformerEncoderBlock(EMBED_DIM, FF_DIM, 1)
    encoder_wrapper = EncoderModel(encoder_block)
    encoder_wrapper(tf.random.normal((1, 49, 1280)))  # Force build
    encoder_wrapper.load_weights('ModelFiles/encoder_wrapper.weights.h5')

    decoder_block = TransformerDecoderBlock(EMBED_DIM, FF_DIM, 2)
    decoder_wrapper = DecoderModel(decoder_block)
    # Force build with sample inputs
    decoder_wrapper([tf.random.normal((1, SEQ_LENGTH-1)), 
                    tf.random.normal((1, 49, EMBED_DIM)),
                    tf.random.normal((1, SEQ_LENGTH-1))])
    decoder_wrapper.load_weights('ModelFiles/decoder_wrapper.weights.h5')
    
    # Load vocabulary
    with open('ModelFiles/vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    
    # ADD THESE LINES:
    print("\n=== Vocabulary Check ===")
    print(f"Total vocab size: {len(vocab)}")
    print("First 20 tokens:", vocab[:20])
    print("Last 20 tokens:", vocab[-20:])
    print("Sample tokens:", [vocab[i] for i in [0, 100, 1000, -1]])


    # print("\n=== Model Architecture Check ===")
    # print("CNN Model:")
    # cnn_model.summary()
    # print("\nEncoder Model:")
    # encoder_wrapper.summary() 
    # print("\nDecoder Model:")
    # decoder_wrapper.summary()


    # Recreate vectorization layer
    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQ_LENGTH,
        standardize=None,
        encoding='utf-8'
    )
    vectorizer.set_vocabulary(vocab)

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def generate_caption(image_path):
    # Preprocess image
    img = preprocess_image(image_path)


    print("\n=== Pipeline Debug ===")
    print("Raw image shape:", img.shape)

    img_for_model = tf.expand_dims(img, 0)
    print("Model input shape:", img_for_model.shape)

# Test CNN features
    features = cnn_model.predict(img_for_model)
    print("CNN features shape:", features.shape)

# Test encoder
    encoded_img = encoder_wrapper(features, training=False)
    print("Encoded features shape:", encoded_img.shape)

    img = tf.expand_dims(img, 0)
    
    # Get image features
    img_features = cnn_model(img)
    
    # Encode features
    encoded_img = encoder_wrapper(img_features, training=False)
    
    # Initialize caption
    decoded_caption = "<start> "
    
    for i in range(SEQ_LENGTH - 1):
        tokenized_caption = vectorizer([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        
        predictions = decoder_wrapper(
            [tokenized_caption, encoded_img, mask], 
            training=False
        )
        
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = vectorizer.get_vocabulary()[int(sampled_token_index)]  # Convert to int
        
        if sampled_token == "<end>":
            break
            
        decoded_caption += " " + str(sampled_token)  # Explicit string conversion
    
    # Clean caption
    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()

    decoded_caption = decoded_caption.encode('utf-8').decode('utf-8')
    
    return decoded_caption

@bp_imageCaption.route('/imagecaptions', methods=['GET', 'POST'])
def upload_file():
    UPLOAD_FOLDER = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    if request.method == 'POST':
        # ✅ Check user ID in session
        user_id = session.get('id')  # Make sure you store user ID in session at login!
        if not user_id:
            return "User not logged in. Please login to upload.", 403

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # ✅ Save to DB with user ID
            connection = get_db_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    insert_query = """
                    UPDATE projectusers
                    SET uploadeduserpic = %s
                    WHERE id = %s
                """
                    cursor.execute(insert_query, (filename,user_id))
                    connection.commit()
                    cursor.close()
                except Exception as e:
                    print(f"Database Insert Error: {e}")
                finally:
                    connection.close()

            # Assuming you have these functions
            caption = generate_caption(filepath)
            image_base64 = generate_caption_image(filepath, caption)

            return render_template('analysis.html',
                                   image_data=image_base64,
                                   caption=caption,
                                   filename=filename)

    return render_template('analysis.html')


if __name__ == '__main__':
    # Load models when starting the app
    load_models()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    bp_imageCaption.run(debug=True)


