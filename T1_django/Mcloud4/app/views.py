"""
Definition of views.
"""

from django.shortcuts import render
from django.http import HttpRequest
from django.http import HttpResponse
from django.template import RequestContext
from datetime import datetime

def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        context_instance = RequestContext(request,
        {
            'title':'Home Page',
            'year':datetime.now().year,
        })
    )

def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        context_instance = RequestContext(request,
        {
            'title':'Contact',
            'message':'Your contact page.',
            'year':datetime.now().year,
        })
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        context_instance = RequestContext(request,
        {
            'title':'About',
            'message':'Your application description page.',
            'year':datetime.now().year,
        })
    )


from app.forms import BootstrapCurveFittingForm
def T1LL_input(request):
    a =1 +1

    return render(
        request,
        'app/fitting_input.html',
        context_instance = RequestContext(request,
        {
            'title':'Fitting Input',
            'form': BootstrapCurveFittingForm,
            'test123':a
               })
    )


def T1LL_result(request):
    crime=request.POST.get('crime')
    zn=request.POST.get('zn')
    inidus=request.POST.get('inidus')
    nox=request.POST.get('nox')
    rm=request.POST.get('rm')
    age=request.POST.get('age')
    dis=request.POST.get('dis')
    rad=request.POST.get('rad')
    tax=request.POST.get('tax')
    ptratio=request.POST.get('ptratio')
    Bk=request.POST.get('Bk')
    lstat=request.POST.get('lstat')
    optradio=request.POST.get('optradio')

    MEDV = bonston_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)
    pic = plot(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)
    result_dict={
        "MEDV":MEDV,
        "pic":pic
    }
    #lr.predict([crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat])
    return render(
    request,
    'app/boston_result.html',
    context_instance = RequestContext(request, result_dict)       )
    #return HttpResponse(y)

def bonston_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat):
    from django.conf import settings
    import os
    import numpy as np
    from sklearn.externals import joblib

    lr=joblib.load(os.path.join(settings.PROJECT_ROOT,'app','lrmachine.pkl'))
    #my_variable =[0.02729,0,7.07,0,0.469,7.185,61.1,4.9671,2,242,17.8,39.283,9.14]
    #Y=lr.predict(np.array(my_variable))

    if not crime:
        my_variable =[0.02729,0,7.07,0,0.469,7.185,61.1,4.9671,2,242,17.8,39.283,9.14]
        Y=lr.predict(np.array(my_variable))
    else:
        x = np.array([crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat], dtype=np.float64)
        Y=lr.predict(x)

    return Y

def plot(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat):
    from django.conf import settings
    import os, matplotlib.pyplot as plt
    from sklearn.externals import joblib
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO
    from sklearn import datasets
    from sklearn.cross_validation import cross_val_predict
    import numpy as np

    lr=joblib.load(os.path.join(settings.PROJECT_ROOT,'app','lrmachine.pkl'))
    boston = datasets.load_boston()
    y = boston.target
    Y=bonston_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)

    try:
        predicted = cross_val_predict(lr, boston.data, y, cv=10)
        predict_y=Y

        plt.figure(1)
        plt.clf()
        plt.scatter(predicted,y)
        plt.plot(predict_y, predict_y, 'ro')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        fig = plt.gcf()
        fig.set_size_inches(10,6)
        #plt.gca().axhline(0, color='black', lw=2)
        plt.gca().grid(True)

        #plt.gca().set_axis_bgcolor('white')
        rv = StringIO()
        plt.savefig(rv, format="svg")
        return rv.getvalue()
    finally:
        plt.clf()
