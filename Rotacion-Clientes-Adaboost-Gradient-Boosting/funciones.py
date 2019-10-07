import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def vals_report(class_rep):
    f1_0= class_rep['No Fuga'][0]
    f1_1= class_rep['Fuga'][0]
    f1_avg= class_rep['weighted avg'][0]
    prec_0= class_rep['No Fuga'][1]
    prec_1= class_rep['Fuga'][1]
    prec_avg= class_rep['weighted avg'][1]
    rec_0= class_rep['No Fuga'][2]
    rec_1= class_rep['Fuga'][2]
    rec_avg= class_rep['weighted avg'][2]
    return f1_0, f1_1, f1_avg, prec_0, prec_1, prec_avg, rec_0, rec_1, rec_avg

def model_metrics(class_rep):
    print(class_rep)
    f1_1, f1_2, f1_3, prec_1, prec_2, prec_3, rec_1,rec_2, rec_3= vals_report(class_rep)
    
    plt.figure(figsize=(10,6))
    
    plt.subplot(1,3,1)
    plt.barh(['Promedio','No Fuga', 'Fuga'], [f1_3, f1_1, f1_2])
    plt.title("F1");
    
    plt.subplot(1,3,2)
    plt.barh(['Promedio','No Fuga', 'Fuga'], [prec_3, prec_1, prec_2])
    plt.title("Precision");
    
    plt.subplot(1,3,3)
    plt.barh(['Promedio','No Fuga', 'Fuga'], [rec_3, rec_1, rec_2])
    plt.title("Recall");
    
    plt.tight_layout()


def plot_feature_importance(fit_model, feat_names):
    """
    Plot relative importance of a feature subset given a fitted model.
    """
    # infer feature importance score
    tmp_importance = fit_model.feature_importances_
    # sort features
    sort_importance = np.argsort(tmp_importance)[::-1]
    # associate feat_names with its relative importance
    names = [feat_names[i] for i in sort_importance]
    # plot
    plt.barh(
        # given range and features
        range(len(feat_names)), tmp_importance[sort_importance]
    )
    # add axis labels identifying attribute name
    plt.yticks(range(len(feat_names)), 
               names, rotation=0)

def infer_k_features(df, model, feat_names, k_feats=10):
    """TODO: Docstring for infer_k_features.
    :arg1: TODO
    :returns: TODO
    """
    # preserve temporary copy
    tmp_df = df.copy()
    # infer feature importance score
    tmp_importance = model.feature_importances_
    # sort features
    sort_importance = np.argsort(tmp_importance)[::-1]
    # associate feat names with its relative importance
    names = [feat_names[i] for i in sort_importance]
    #mungle into a dataframe
    tmp_attr = pd.DataFrame(
        {'name': names,
         'score': tmp_importance[sort_importance]}
    )
    # restrict dataframe to k attributes
    tmp_attr = tmp_attr[:k_feats]['name']
    
    list_attr=[]
    for n in tmp_attr:
        list_attr.append(n)

    # filter attributes
    #tmp_df = tmp_df[tmp_df.columns[tmp_df.columns.isin(tmp_attr)]]
    tmp_df = tmp_df.loc[:,list_attr]
    

    return tmp_df
