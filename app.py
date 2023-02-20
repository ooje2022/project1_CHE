from flask import Flask, request, render_template
import joblib
import numpy as np
import json

model = joblib.load('model_svr_21_02_2023_00-11-34.joblib')

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/house-price/')
def house_price():
    return render_template('house_price.html')

@app.route('/house-price/predict/', methods=["POST"])
def predict():

    if request.method == 'POST':
        medInc = float(request.form['medInc']) / 100000
        houseAge = float(request.form['houseAge'])
        aveRooms = float(request.form['aveRooms'])
        aveBedrms = float(request.form['aveBedrms'])
        population = float(request.form['population'])
        aveOccup = float(request.form['aveOccup'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        input_vec = [medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude]
        data = np.array(input_vec).reshape(1, -1)

        submission = request.form

        y_pred = model.predict(data)[0] *100000

        pred_price = "${:,.2f}".format(y_pred)

        print("Estimated price: ", pred_price)

        dict = {
            "result": pred_price,
            "submission": submission
        }
        
        return render_template('house_price.html', result = pred_price)


"""


def predict(event, context):
    body = {
        "message": "Ok",
      
    }

    if 'queryStringParameters' in event.keys():
        params = event['queryStringParameters']

        medInc = float(params['medInc']) / 100000
        houseAge = float(params['houseAge'])
        aveRooms = float(params['aveRooms'])
        aveBedrms = float(params['aveBedrms'])
        population = float(params['population'])
        aveOccup = float(params['aveOccup'])
        latitude = float(params['latitude'])
        longitude = float(params['longitude'])

        input_vector = [medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude]
        
        y_pred = model.predict([input_vector])[0] * 100000 
        y_pred = round(y_pred, 2)

        body['pred_price'] = y_pred

    else:
        body['message'] = 'Incomplete or missing parameters in event.'

    print(body['message'])

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Access-Control-Allow-Origin": "*"
        }
    }

    return render_template('get_price.html', response)

    #return response

def do_main():
    event = {
        'queryStringParameters': {
            'medInc': 200000,
            'houseAge': 10,
            'aveRooms': 4,
            'aveBedrms': 1,
            'population': 800,
            'aveOccup': 3,
            'latitude': 37.54,
            'longitude': -121.72
        }
    }

    response = predict(event, None) # None is the context
    body = json.loads(response['body'])
    print('Price:', "${:,.2f}".format(body['pred_price']))

    with open('event.json', 'w') as event_file:
        event_file.write(json.dumps(event))
    

#do_main()
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)