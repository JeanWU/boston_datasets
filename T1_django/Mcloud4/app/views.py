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
    #import pyFitMR.Fitting_lib as FB
    from django.conf import settings
    import os
    #print(settings.PROJECT_ROOT)
    import numpy

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

    x = numpy.array([crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat], dtype=numpy.float64)
    print(ptratio)
    from sklearn.externals import joblib
    lr=joblib.load(os.path.join(settings.PROJECT_ROOT,'app','lrmachine.pkl'))
    y=lr.predict(x)
    #lr.predict([crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat])
    return render(        request,        'app/fitting_result.html',        context_instance = RequestContext(request, fitted_result_dict)       )
    #return HttpResponse(y)
