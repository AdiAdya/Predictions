import tensorflow as tf
import tensorflow_hub as hub
Model = tf.keras.models.load_model('my_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})

# main function
def predict(query):   
    prediction = Model.predict([query])
    print(prediction)
    return prediction

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = predict(...)
#   print(result)
