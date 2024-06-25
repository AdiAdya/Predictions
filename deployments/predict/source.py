import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
Model = tf.keras.models.load_model('data/Model.h5',custom_objects={'KerasLayer':hub.KerasLayer})


def predict(query):   
    prediction = Model.predict([query])
    prediction = 'Classified' if prediction[0] > 0.5 else 'Benign'
    return prediction


# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = predict(...)
#   print(result)
