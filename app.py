import pandas as pd
import joblib
from flask import (
     Flask,
     url_for,
     render_template
)
from forms import InputForm

model = joblib.load("model.joblib")

app = Flask(__name__)
app.config["SECRET_KEY"]="secret_key"

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html",title="HOME")

@app.route("/predict",methods=["GET","POST"])
def predict():
     form = InputForm()
     if form.validate_on_submit():
          x_new = pd.DataFrame(dict(
               airline = [form.airline.data],
               date_of_journey = [form.date_of_journey.data.strftime("%Y-%m-%d")],
               source = [form.source.data],
               destination = [form.destination.data],
               dep_time = [form.dep_time.data.strftime("%H:%M:%S")],
               arrival_time = [form.arrival_time.data.strftime("%H:%M:%S")],
               duration = [form.duration.data],
               total_stops = [form.total_stops.data],
               additional_info = [form.additional_info.data],
          ))
          prediction = model.predict(x_new)[0]
          message = f"Predicted Value is {prediction:,.0f} INR"
     else:
          message = "Please provide valid details"     
     return render_template("predict.html",title="PREDICT",form=form,output=message)

if __name__ == "__main__":
     app.run(debug=True)