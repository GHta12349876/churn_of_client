from datetime import datetime
import io
import os
import pandas as pd
from flask import Flask,jsonify,render_template,request,send_file
from model.model_service import get_default_service

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=16*1024*1024

prediction_service = get_default_service()
CURRENT_MODEL = 'original'

def predict_churn(data,model='original'):
    return prediction_service.predict_churn(data,model=model)

def get_risk_level(probability,model='original'):
    return prediction_service._get_risk_level(probability,model=model)

def get_recommendations(risk_level):
    return prediction_service.get_recommendations(risk_level)

def process_batch_data(df,model='original'):
    return prediction_service.process_batch_data(df,model=model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        monthly_charges = request.form.get('monthly_charges')
        if monthly_charges not in (None,'') and float(monthly_charges)<0:
            return jsonify({'error':'MonthlyCharges must be non-negative'}),400

        data = {
            'SeniorCitizen':int(request.form['senior_citizen']),
            'tenure':int(request.form['tenure']),
            'PaymentMethod':request.form['payment_method'],
            'InternetService':request.form['internet_service'],
            'Contract':request.form['contract']
        }
        if data['tenure']<0:
            return jsonify({'error': 'tenure must be non-negative'}),400
        if data['SeniorCitizen']<0:
            return jsonify({'error': 'tenure must be non-negative'}),400

        probability = float(predict_churn(data,model=CURRENT_MODEL)[0])
        risk_level = get_risk_level(probability,model=CURRENT_MODEL)
        recommendations = get_recommendations(risk_level)

        return  jsonify({
            'success':True,
            'probability':round(probability*100,2),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'model_used': CURRENT_MODEL
        })
    except Exception as e:
        return jsonify({'error':str(e)}),400


@app.route('/batch_predict',methods=['POST'])  # 批量
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error':'No file provide'}),400

        file = request.files['file']
        if file.filename=='':
            return jsonify({'error':'No file selected'}),400
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx','.xls')):
            df= pd.read_excel(file)
        else:
            return jsonify({'error':'Unsupported file format.Please upload CSV or Excel file.'}),400

        result_df= process_batch_data(df,model=CURRENT_MODEL)
        results = []
        for _,row in result_df.iterrows():
            monthly_charges = row.get('MonthlyCharges',None)
            if pd.isna(monthly_charges):
                monthly_charges=None
            results.append({
                'row': int(row.name) + 1,  # 很奇怪但又很标准的写法
                'monthly_charges': float(monthly_charges) if monthly_charges is not None else None,
                'senior_citizen': int(row['SeniorCitizen']),
                'tenure': int(row['tenure']),
                'payment_method': str(row['PaymentMethod']),
                'internet_service': str(row['InternetService']),
                'contract': str(row['Contract']),
                'probability': round(float(row['ChurnProbability']) * 100, 2),
                'risk_level': str(row['RiskLevel'])
            })
        app.config['LAST_BATCH_RESULTS'] = result_df

        return jsonify({
            'success':True,
            'count':len(results),
            'results':results,
            'model_uesd':CURRENT_MODEL
        })

    except Exception as e:
        return jsonify({'error':str(e)}),400

@app.route('/export',methods=['POST'])
def export_results():
    try:
        if 'LAST_BATCH_RESULTS' not in app.config or app.config['LAST_BATCH_RESULTS'] is None:
            return jsonify({'error':'No results to export. Please run batch prediction first.'}),400
        df = app.config['LAST_BATCH_RESULTS'].copy()
        df['Recommendations']=df['RiskLevel'].apply(lambda x:'; '.join(get_recommendations(x)))
        df['ChurnProbability']=df['ChurnProbability'].apply(lambda x:f'{x*100:.2f}%')

        # 把数据变成excel文件，但不存到硬盘上，而是内存里，用完即删
        output = io.BytesIO()
        with pd.ExcelWriter(output,engine='openpyxl') as writer:
            df.to_excel(writer,sheet_name='Churn Predictions',index=False)
        output.seek(0)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'churn_predictions_{timestamp}.xlsx'

        return send_file(
            output, # 文件在哪里
            mimetype='application/vnd.openxmlformats-officedocument .spreadsheetml.sheet',  # 文件是什么类型，不用记，这是告诉浏览器的
            as_attachment=True, # 是否下载
            download_name=filename
        )

    except Exception as e:
        return jsonify({'error':str(e)}),400

@app.route('/model',methods=['GET','POST'])
def model_config():
    global CURRENT_MODEL # 必须要声明，否则下面的重定义就是局部的了
    if request.method=='GET':
        metadata = prediction_service.get_metadata()
        return jsonify(metadata)

    model = request.json.get('model','original')
    if model not in ['original','rebuilt']:
        return jsonify({'error':'Invalid model. '}),400

    CURRENT_MODEL=model
    return jsonify({
        'success':True,
        'current_model':CURRENT_MODEL,
        'message':f'Switched to {model} model'
    })

if __name__=='__main__':
    port = int(os.environ.get('PORT',5001))
    app.run(debug=True,port=port,threaded=True)





