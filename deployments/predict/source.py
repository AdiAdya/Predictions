import modelbit, sys
from typing import *
from tensorflow import keras


Model = keras.models.load_model('data/Model.h5')

# main function
async def predict(query):   
    prediction = await Model.predict([query])
    print(prediction)
    return prediction

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   import asyncio
#   result = asyncio.run(predict(...))
#   print(result)