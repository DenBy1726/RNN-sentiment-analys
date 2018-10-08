from flask import Flask,request,jsonify
import train
import numpy as np
from keras.preprocessing.sequence import pad_sequences

tokenizer = train.load_tokenizer("./model/tokenizer.pickle")
model = train.load_model("./model/model.json", "./model/weights.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return str(model.to_json()) 


@app.route('/', methods=['POST'])
def create_task():
    response = {}
    response["text"] = request.data.decode("utf-8")

    twt = tokenizer.texts_to_sequences([response["text"]])
    twt = pad_sequences(twt,maxlen = 753,dtype='int32', value=0)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
    if(np.argmax(sentiment) == 0):
        response["result"] = "negative"
    elif (np.argmax(sentiment) == 1):
        response["result"] = "positive"
    return jsonify(response)

app.run(debug=True)