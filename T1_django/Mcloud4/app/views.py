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
        #'app/index.html',
        'app/fitting_input.html',
        context_instance = RequestContext(request,
        {
            'title':'Property Value Prediction',
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
            #'message':'Your contact page.',
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
            'message':'This is a website providing you the prediction of your property value.',
            'year':datetime.now().year,
        })
    )


from app.forms import BootstrapCurveFittingForm
def T1LL_input(request):

    return render(
        request,
        'app/fitting_input.html',
        context_instance = RequestContext(request,
        {
            'title':'Fitting Input',
            'form': BootstrapCurveFittingForm
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

    MEDV_linear = linear_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)
    pic_linear = linear_plot(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)
    MEDV_SVR = SVR_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)
    pic_SVR = SVR_plot(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)
    script, div = bokeh_plot(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)

    result_dict={
        "MEDV_linear":MEDV_linear,
        "pic_linear":pic_linear,
        "MEDV_SVR":MEDV_SVR,
        "pic_SVR":pic_SVR,
        "the_script":script,
        "the_div":div
    }
    return render(
    request,
    'app/boston_result.html',
    context_instance = RequestContext(request, result_dict)       )
    #return HttpResponse(y)

def linear_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat):
    from django.conf import settings
    import os
    import numpy as np
    from sklearn.externals import joblib

    lr=joblib.load(os.path.join(settings.PROJECT_ROOT,'app','lrmachine.pkl'))

    if not crime:
        my_variable =[0.02729,0,7.07,0,0.469,7.185,61.1,4.9671,2,242,17.8,39.283,9.14]
        Y=lr.predict(np.array(my_variable))
    else:
        x = np.array([crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat], dtype=np.float64)
        Y=lr.predict(x)

    return Y

def linear_plot(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat):
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
    Y=linear_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)

    try:
        predicted = cross_val_predict(lr, boston.data, y, cv=10)
        predict_y=Y

        plt.figure(1)
        plt.clf()
        plt.scatter(predicted,y,s=2)
        plt.plot(predict_y, predict_y, 'ro')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        fig = plt.gcf()
        fig.set_size_inches(10,6)
        plt.gca().grid(True)

        rv = StringIO()
        plt.savefig(rv, format="svg")
        return rv.getvalue()
    finally:
        plt.clf()

def SVR_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat):
    from django.conf import settings
    import os
    import numpy as np
    from sklearn.externals import joblib

    clf=joblib.load(os.path.join(settings.PROJECT_ROOT,'app','machine_SVR.pkl'))

    if not crime:
        my_variable =[0.02729,0,7.07,0,0.469,7.185,61.1,4.9671,2,242,17.8,39.283,9.14]
        Y=clf.predict(np.array(my_variable))
    else:
        x = np.array([crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat], dtype=np.float64)
        Y=clf.predict(x)

    return Y

def SVR_plot(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat):
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

    clf=joblib.load(os.path.join(settings.PROJECT_ROOT,'app','machine_SVR.pkl'))
    boston = datasets.load_boston()
    y = boston.target
    Y=SVR_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)

    try:
        predicted = clf.predict(boston.data)
        predict_y=Y

        plt.figure(1)
        plt.clf()
        plt.scatter(predicted,y,s=2)
        plt.plot(predict_y, predict_y, 'ro')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        fig = plt.gcf()
        fig.set_size_inches(10,6)
        plt.gca().grid(True)

        rv = StringIO()
        plt.savefig(rv, format="svg")
        return rv.getvalue()
    finally:
        plt.clf()


def bokeh_plot(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat):
    from django.conf import settings
    import os
    from sklearn.externals import joblib
    from sklearn import datasets
    from bokeh.plotting import figure, show, output_file
    from bokeh.resources import CDN
    from bokeh.embed import components

    clf=joblib.load(os.path.join(settings.PROJECT_ROOT,'app','machine_SVR.pkl'))
    boston = datasets.load_boston()
    y = boston.target
    Y=SVR_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat)


    predicted = clf.predict(boston.data)
    predict_y=Y
    p = figure(title = "Boston dataset")
    p.xaxis.axis_label = 'Measured'
    p.yaxis.axis_label = 'Predicted'

    p.scatter(y,predicted)
    p.asterisk(x=predict_y, y=predict_y, size=20, color="#F0027F")
    script, div = components(p, CDN)
    return script, div
