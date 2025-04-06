#importing necessary libraries
import os
import time
import sys
import socket
from flask import Flask, request, render_template
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader
import weka.core.serialization as serialization
from weka.core.dataset import Instance
from weka.core.classes import Random
import csv
import json
from datetime import datetime

# Initializing Flask App
app = Flask(__name__)

# Dataset Path
dataset_path = os.path.join(os.getcwd(), "model", "SaYoPillow.arff")

# Server Loading Message
print("Please wait! The server is starting...")
for i in range(1, 101):
    time.sleep(0.01)
    sys.stdout.write(f"\rLoading: {i}%")
    sys.stdout.flush()
print("\nServer is ready to use! \n")

# Start JVM
jvm.start()

# Loading Dataset
try:
    loader = Loader(classname="weka.core.converters.ArffLoader")
    dataset = loader.load_file(dataset_path)
    dataset.class_is_last()
    print("\nDataset Loaded Successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = None

# Prediction history file
history_file = os.path.join(os.getcwd(), "prediction_history.csv")
if not os.path.exists(history_file):
    with open(history_file, mode='w', newline='') as file:
        pass

# Model Names
model_names = {
    "random_committee": "Random Committee",
    "hoeffding_tree": "Hoeffding Tree",
    "logit_boost": "Logit Boost",
    "rep_tree": "Rep Tree",
    "bagging": "Bagging",
    "lwl": "Lwl"
}

# Loading WEKA models
models = {}
for key in model_names:
    model_file = os.path.join(os.getcwd(), "model", f"{key}.model")
    try:
        loaded_model = serialization.read(model_file)
        models[key] = Classifier(jobject=loaded_model)
        print(f"{model_names[key]} model loaded successfully!")
    except Exception as e:
        print(f"Error loading {model_names[key]}: {e}")

# Generate model metrics (accuracy, precision, recall, f_measure)
metrics = {}
if dataset is not None:
    for key, clf in models.items():
        try:
            evaluation = Evaluation(dataset)
            evaluation.crossvalidate_model(clf, dataset, 10, Random(1))
            metrics[model_names[key]] = {
                "accuracy": f"{evaluation.percent_correct:.2f}%",
                "precision": f"{evaluation.weighted_precision:.2f}",
                "recall": f"{evaluation.weighted_recall:.2f}",
                "f_measure": f"{evaluation.weighted_f_measure:.2f}"
            }
        except Exception as e:
            print(f"Evaluation error for {key}: {e}")

# Save metrics to JSON file
metrics_json_path = os.path.join(os.getcwd(), "model_metrics.json")
try:
    with open(metrics_json_path, "w") as json_file:
        json.dump(metrics, json_file, indent=4)
    print("Model evaluation metrics saved to metrics.json!")
except Exception as e:
    print(f"Error saving metrics to JSON: {e}")

# Save prediction to CSV
def log_prediction(model_name, input_values, stress_label):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(history_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, model_name, input_values, stress_label])

# default route
@app.route('/')
def home():
    return render_template('prediction.html')

# route to prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if not models or dataset is None:
        return render_template('prediction.html', prediction_text="Error: Models or Dataset not loaded!", precaution_text="Please check your setup.")

    try:
        selected_model_key = request.form.get("model")
        chosen_model = models.get(selected_model_key)

        if chosen_model is None:
            return render_template('prediction.html', prediction_text="Error: Model not found!", precaution_text="Invalid model selected.")

        input_values = [float(x) for x in list(request.form.values()) if x.replace('.', '', 1).isdigit()]
        instance = Instance.create_instance(input_values)
        instance.dataset = dataset
        result = int(chosen_model.classify_instance(instance))

        stress_levels = {
            0: ("Low/Normal", "You're doing fine! Just eat well, sleep properly, and stay active."),
            1: ("Medium Low", "Relax a bit—try meditation, drink more water, and avoid too much tea or coffee."),
            2: ("Medium", "Take small breaks, go for a walk, and reduce screen time."),
            3: ("Medium High", "Follow a proper sleep routine, try to stay calm, and talk to someone if needed."),
            4: ("High", "This is serious. Get help, take care of yourself, and don’t overwork."),
        }

        stress_label, precaution = stress_levels.get(result, ("Unknown", "No precaution available."))
        log_prediction(model_names[selected_model_key], input_values, stress_label)

        return render_template('prediction.html', prediction_text=stress_label, precaution_text=precaution)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template('prediction.html', prediction_text="Prediction Error!", precaution_text="Something went wrong. Try again.")

# route to history page
@app.route('/history')
def history():
    records = []
    if os.path.exists(history_file):
        with open(history_file, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 4:
                    timestamp, model, input_values, stress_label = row
                    records.append({
                        "timestamp": timestamp,
                        "model": model,
                        "input_values": input_values,
                        "stress_label": stress_label
                    })
    return render_template('history.html', records=records, metrics=metrics)

# Running App
if __name__ == "__main__":   # Start the Flask app
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        jvm.stop()    # Stop the JVM when the app is closed
