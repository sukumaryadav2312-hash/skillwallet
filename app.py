from flask import Flask, render_template, request, jsonify
import joblib, pandas as pd, json, os

# ensure template dir and default page exist
os.makedirs('templates', exist_ok=True)
if not os.path.exists(os.path.join('templates','index.html')):
    with open(os.path.join('templates','index.html'),'w') as f:
        f.write("""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hypertension Risk Assessment</title>
    <style>
        body {font-family: Arial, sans-serif; margin:0; padding:0; background:#f4f6f8; color:#333;}
        header {background:#2a9d8f; color:#fff; padding:20px;text-align:center;}
        .container {max-width:900px;margin:30px auto;background:#fff;padding:20px;box-shadow:0 0 10px rgba(0,0,0,0.1);}
        .form-grid {display:grid;grid-template-columns:1fr 1fr;gap:15px;}
        label {display:block;margin-bottom:5px;font-weight:bold;}
        select,input {width:100%;padding:8px;border:1px solid #ccc;border-radius:4px;}
        button {background:#e76f51;color:#fff;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;font-size:16px;}
        button:hover {background:#d65a3e;}
        .result {margin-top:20px;padding:15px;background:#ade8f4;border-left:5px solid #00b4d8;}
    </style>
</head>
<body>
<header>
    <h1>Hypertension Detection</h1>
    <p>Advanced AI Cardiovascular Risk Assessment</p>
</header>
<div class="container">
    <form id="riskForm">
        <div class="form-grid">
            <div><label for="gender">Gender</label><select id="gender" name="Gender"><option value="">Select</option><option>Male</option><option>Female</option></select></div>
            <div><label for="age">Age Group</label><select id="age" name="Age"><option value="">Select</option><option>18-34</option><option>35-50</option><option>51-64</option><option>65+</option></select></div>
            <div><label for="history">Family History</label><select id="history" name="History"><option value="">Select</option><option>Yes</option><option>No</option></select></div>
            <div><label for="medication">Under Medical Care</label><select id="medication" name="TakeMedication"><option value="">Select</option><option>Yes</option><option>No</option></select></div>
            <div><label for="severity">Symptom Severity</label><select id="severity" name="Severity"><option value="">Select</option><option>Mild</option><option>Moderate</option><option>Severe</option></select></div>
            <div><label for="breath">Shortness of Breath</label><select id="breath" name="BreathShortness"><option value="">Select</option><option>Yes</option><option>No</option></select></div>
            <div><label for="vision">Vision Changes</label><select id="vision" name="VisualChanges"><option value="">Select</option><option>Yes</option><option>No</option></select></div>
            <div><label for="nosebleed">Nosebleeds</label><select id="nosebleed" name="NoseBleeding"><option value="">Select</option><option>Yes</option><option>No</option></select></div>
            <div><label for="systolic">Systolic Pressure</label><select id="systolic" name="Systolic"><option value="">Select</option><option>90 - 100</option><option>101 - 110</option><option>111 - 120</option><option>121 - 130</option><option>131 - 140</option><option>141 - 150</option><option>151 - 160</option></select></div>
            <div><label for="diastolic">Diastolic Pressure</label><select id="diastolic" name="Diastolic"><option value="">Select</option><option>51 - 60</option><option>61 - 70</option><option>70 - 80</option><option>81 - 90</option><option>91 - 100</option></select></div>
            <div><label for="diet">Heart-Healthy Diet</label><select id="diet" name="ControlledDiet"><option value="">Select</option><option>Yes</option><option>No</option></select></div>
            <div><label for="diagnosis">Time Since Diagnosis</label><select id="diagnosis" name="Whendiagnoused"><option value="">Select</option><option>Less than 1 Year</option><option>1-3 Years</option><option>Over 3 Years</option></select></div>
        </div>
        <p style="text-align:center;"><button type="button" onclick="submitForm()">Generate Risk Assessment</button></p>
    </form>
    <div id="output" class="result"></div>
</div>
<script>
function submitForm(){
    const form = document.getElementById('riskForm');
    const data = {};
    new FormData(form).forEach((v,k)=> data[k]=v);
    fetch('/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})
    .then(r=>r.json()).then(res=>{
        document.getElementById('output').innerHTML =
            `<h3>Prediction: ${res.stage} (Risk ${res.risk}%)</h3>` +
            `<p>Urgency: ${res.urgency}</p><pre>${JSON.stringify(res.probs,null,2)}</pre>`;
    });
}
</script>
</body>
</html>""")


app = Flask(__name__)

# load artifacts, with clear error if missing
try:
    model = joblib.load('best_hypertension_model.pkl')
except FileNotFoundError:
    raise RuntimeError(
        "Model file 'best_hypertension_model.pkl' not found. "
        "Please run Hypertension_Prediction_System.py to train and save the model first."
    )

try:
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    raise RuntimeError(
        "Label encoder 'label_encoder.pkl' not found. "
        "Ensure the training script has been executed."
    )
with open('model_metadata.json') as f:
    metadata = json.load(f)

@app.route('/')
def home():
    # dashboard landing page
    return render_template('index.html', metadata=metadata)

# use a less common endpoint
@app.route('/evaluate', methods=['POST'])
def evaluate():
    # parse JSON payload coming from client
    input_data = request.get_json()
    # validate that critical fields are not empty
    required = ['Gender','Age','History','Patient','TakeMedication','Severity',
                'BreathShortness','VisualChanges','NoseBleeding','Systolic','Diastolic',
                'ControlledDiet','Whendiagnoused']
    missing = [k for k in required if not input_data.get(k)]
    if missing:
        return jsonify({'error': 'Missing input fields', 'fields': missing}), 400
    df_row = pd.DataFrame([input_data])
    raw_pred = model.predict(df_row)[0]
    pred_stage = label_encoder.inverse_transform([raw_pred])[0]

    # build probability breakdown
    probabilities = model.predict_proba(df_row)[0]
    prob_map = {
        label_encoder.inverse_transform([idx])[0]: round(float(val), 4)
        for idx, val in enumerate(probabilities)
    }
    confidence = round(float(max(probabilities) * 100), 1)

    # basic urgency assessment
    st = str(pred_stage).lower()
    if 'normal' in st or 'pre' in st:
        urgency = 'LOW'
        recommendations = [
            'Maintain balanced diet',
            'Exercise regularly',
            'Monitor BP weekly'
        ]
    elif 'stage-1' in st or 'stage 1' in st or st == '1':
        urgency = 'MODERATE'
        recommendations = [
            'Start lifestyle changes',
            'Adopt DASH eating plan',
            'Track BP daily'
        ]
    elif 'stage-2' in st or 'stage 2' in st or st == '2':
        urgency = 'HIGH'
        recommendations = [
            'See a doctor ASAP',
            'Medication likely',
            'Frequent BP checks'
        ]
    else:
        urgency = 'CRITICAL'
        recommendations = [
            'Immediate medical attention required',
            'Follow prescribed treatment',
            'Use continuous monitoring'
        ]

    return jsonify({
        'stage': pred_stage,
        'risk': confidence,
        'probs': prob_map,
        'urgency': urgency,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)