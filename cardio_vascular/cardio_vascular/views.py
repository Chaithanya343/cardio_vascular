import numpy as np
from django.http import HttpResponse
from django.shortcuts import render
import pickle
def getstarted(request):
    return render(request,"getstarted.html")
def home(request):
    return render(request,"home.html")

def getpredections(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
    model=pickle.load(open('model_pickle','rb'))
    input_data = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        return 'no'
    else:
        return "yes"
def output(request):
    # data = {}
    pred=''
    try:
        if request.method == 'POST':
            age = request.POST.get('age', None)
            sex_ip = request.POST.get('sex', None)
            if sex_ip=="Male":
                sex=1
            else:
                sex=0
            cp = request.POST.get('cp', None)
            trestbps = request.POST.get('trestbps', None)
            chol = request.POST.get('chol', None)
            fbs = request.POST.get('fbs', None)
            restecg = request.POST.get('restecg', None)
            thalach = request.POST.get('thalach', None)
            exang = request.POST.get('exang', None)
            oldpeak = request.POST.get('oldpeak', None)
            slope = request.POST.get('slope', None)
            ca = request.POST.get('ca', None)
            thal = request.POST.get('thal', None)

            pred=getpredections(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
    except Exception as e:

        print(f"An error occurred: {e}")
        # data["error_message"] = "An error occurred while processing the form data."
    return render(request, "output.html",{"res":pred})
