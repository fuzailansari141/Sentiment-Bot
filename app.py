from flask import *
import joblib 

app=Flask(__name__)

vect = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

@app.route("/", methods=['GET', 'POST'])
def home():
    pred = None
    text=None
    if request.method == 'POST':
        text = request.form["predict"]
        new_text = [text]
        new_text_vectorized = vect.transform(new_text)
        new_text_predictions = model.predict(new_text_vectorized)

        for prediction in new_text_predictions:
            if prediction == 1:
                pred = "Positive"
            else:
                pred = "Negative"
            break
    return render_template("home.html", pred=pred,text=text)

if __name__=="__main__":
    app.run(debug=True)