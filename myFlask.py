from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('personality_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('startpage.html')

@app.route('/test')
def istarts():
    return render_template('MyWebGetData.htm')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            yes_no_map = {'Yes': 1, 'No': 0}

            stage_fear_raw = request.form.get('stage_fear', 'No')
            drained_raw = request.form.get('drained', 'No')
            stage_fear_enc = yes_no_map.get(stage_fear_raw, 0)
            drained_enc = yes_no_map.get(drained_raw, 0)

            time_alone = float(request.form.get('time_spent_alone', 0))  
            social_events = float(request.form.get('social_events', 0)) 
            going_out = float(request.form.get('going_outside', 0))   
            friends = float(request.form.get('friends_count', 0))  
            posts = float(request.form.get('post_frequency', 0))         

              #Scale Selected Features
            numerical_features = np.array([[
                time_alone, 
                social_events, 
                going_out, 
                friends, 
                posts
            ]])
            scaled_values = scaler.transform(numerical_features)[0] 

            #Compile Final Features
            final_features = np.array([[
                scaled_values[0],
                stage_fear_enc,
                scaled_values[1], 
                scaled_values[2],
                drained_enc,     
                scaled_values[3],
                scaled_values[4]  
            ]])

            #Prediction
            prediction = model.predict(final_features)[0]
            
            #Determine Result (0=Extrovert, 1 = Introvert)
            if str(prediction) == '0':
                personality_type = "Extrovert"
                desc = "You gain energy from social interactions and enjoy being around people!"
            else:
                personality_type = "Introvert"
                desc = "You feel more energized by spending time alone and prefer quiet environments."

            return render_template('nextpage.html', 
                                   result=personality_type, 
                                   description=desc)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return redirect(url_for('istarts'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)