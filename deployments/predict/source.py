import modelbit, sys
from typing import *
from tensorflow import keras


Model = keras.models.load_model('data/Model.h5')

# main function
def predict(query):   
    prediction = Model.predict([query])
    print(prediction)
    return prediction

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = predict(...)
#   print(result)