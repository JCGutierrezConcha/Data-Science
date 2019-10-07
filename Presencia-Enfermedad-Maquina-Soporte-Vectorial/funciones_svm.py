import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def vals_report(class_rep):
    f1_0= class_rep['Benigno'][0]
    f1_1= class_rep['Maligno'][0]
    f1_avg= class_rep['weighted avg'][0]
    prec_0= class_rep['Benigno'][1]
    prec_1= class_rep['Maligno'][1]
    prec_avg= class_rep['weighted avg'][1]
    rec_0= class_rep['Benigno'][2]
    rec_1= class_rep['Maligno'][2]
    rec_avg= class_rep['weighted avg'][2]
    return f1_0, f1_1, f1_avg, prec_0, prec_1, prec_avg, rec_0, rec_1, rec_avg


def model_metrics(class_rep):
    print(class_rep)
    f1_1, f1_2, f1_3, prec_1, prec_2, prec_3, rec_1,rec_2, rec_3= vals_report(class_rep)
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.barh(['Promedio','Benigno', 'Maligno'], [f1_3, f1_1, f1_2])
    plt.title("F1");
    
    plt.subplot(1,3,2)
    plt.barh(['Promedio','Benigno', 'Maligno'], [prec_3, prec_1, prec_2])
    plt.title("Precision");
    
    plt.subplot(1,3,3)
    plt.barh(['Promedio','Benigno', 'Maligno'], [rec_3, rec_1, rec_2])
    plt.title("Recall");
    
    plt.tight_layout()