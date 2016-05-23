
def bonston_fitting(crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat):
    from django.conf import settings
    import os
    import numpy as np
    from sklearn.externals import joblib
    #from scipy.optimize.minpack import curve_fit
    import pyT1

    lr=joblib.load(os.path.join(settings.PROJECT_ROOT,'app','lrmachine.pkl'))

    if bonston_fitting.my_variable is None:
        bonston_fitting.my_variable =[0.02729,0,7.07,0,0.469,7.185,61.1,4.9671,2,242,17.8,39.283,9.14,1]
        Y=lr.predict(bonston_fitting.my_variable)
    else:
        x = np.array([crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat], dtype=np.float64)
        Y=lr.predict(x)

    return Y

    """if crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk and lstat:
        demo_data = 0
    else: #default value for testing
        crime=0.02729
        zn=0
        inidus=7.07
        optradio=0
        nox=0.469
        rm=7.185
        age=61.1
        dis=4.9671
        rad=2
        tax=242
        ptratio=17.8
        Bk=39.283
        lstat=9.14
        demo_data = 1
"""




"""    x = np.array([crime,zn,inidus,optradio,nox,rm,age,dis,rad,tax,ptratio,Bk,lstat], dtype=numpy.float64)
    dict_T1fit_result =pyT1.bonston_fitting(t_value,y_value)"""

    # prepare HTML_text1 , first block of the result page
    #========================================================================
"""    if dict_T1fit_result ['error_status']:
        result_dict = {    'HTML_text1': "<div class='alert alert-danger' role='alert'> %s </div> " %    dict_T1fit_result ['error_str']  }
        return result_dict

    HTML_text1 =''
    if demo_data:
        HTML_text1 += "<p><p><div class='alert alert-warning' role='alert'>No input data, using demo data set!</div>"

    HTML_text1 +=' <h4>T1 Fitting Model for Look-Locker Experiment</h4>'
    HTML_text1 +="<p> $$ SI =  {A - B e^{- \\frac{TI}{T_{1}^*}}}$$ "  #在兩個$$中間包住LATEX表示法的算式
    HTML_text1 += "The obtained T1 model are: A = %.2f, B = %.2f, T1* = %.2f ms<p>" % (dict_T1fit_result ['A'] , dict_T1fit_result ['B'] , dict_T1fit_result ['T1_star'] )
    HTML_text1 += "<p> $$ SI =  {%.2f - %.2f e^{- \\frac{TI}{%.2f}}}$$" %  (dict_T1fit_result ['A'] , dict_T1fit_result ['B'] , dict_T1fit_result ['T1_star'] )
    HTML_text1 += "<H4>LL corrected T1: %.2f ms</H4><p>" % dict_T1fit_result ['T1']
    #========================================================================

# prepare High Chart , Second block of the result page
#========================================================================
    HTML_text2 = "<p>t values = %s" % t_value
    HTML_text2 += "<p>y values = %s" % y_value
    HTML_text2 +="<p> fitting residue = %.2f<p>" % dict_T1fit_result ['residue']
        #print HTML
    import Plotting_lib
    chart_title = 'T1 fitting for Look-Locker Experiment'
    xAxis_label = 'Inversion Time (ms)'
    yAxis_label = 'Signal Intensity (A.U.)'
    result_dict = {
            'HTML_text1': HTML_text1,
            'HTML_text2': HTML_text2,
            'SVG':Plotting_lib.dynamic_svg(dict_T1fit_result ['t_val_org'],dict_T1fit_result ['y_val_org'],dict_T1fit_result ['t_val_fit'], dict_T1fit_result ['y_val_fit'],xAxis_label,yAxis_label,chart_title),
            'highchart':Plotting_lib.highchart(dict_T1fit_result ['t_val_org'],dict_T1fit_result ['y_val_org'],dict_T1fit_result ['t_val_fit'], dict_T1fit_result ['y_val_fit'],xAxis_label,yAxis_label,chart_title),
               }
    return result_dict"""
