import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
def run_shap_analysis(best_model_pipeline, X, y, feature_names, top_model_name, output_dir):

    try:
        preprocess = best_model_pipeline.named_steps['prep']
        clf = best_model_pipeline.named_steps['clf']
        
        
        X_processed = preprocess.transform(X)
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
       
        if hasattr(clf, 'feature_importances_'):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_processed_df)
            if isinstance(shap_values, list): shap_values = shap_values[1]
        elif hasattr(clf, 'coef_'):
            explainer = shap.LinearExplainer(clf, X_processed_df, feature_dependence="independent")
            shap_values = explainer.shap_values(X_processed_df)
        else:
            explainer = shap.PermutationExplainer(clf.predict_proba, X_processed_df)
            shap_values = explainer.shap_values(X_processed_df)[:,:,1]

        # =========================================================
        #  plot 1
        # =========================================================
        
        plot_values = [] 
        colors = []      
        plot_labels = [] 
        
        y_array = np.array(y)
        
       
        for idx, feat_name in enumerate(feature_names):
            
            s_vals = shap_values[:, idx]     
            f_vals = X_processed[:, idx]     
            
            
            magnitude = np.abs(np.mean(s_vals[y_array==1]) - np.mean(s_vals[y_array==0]))
            
            
            if np.std(f_vals) == 0 or np.std(s_vals) == 0:
                corr = 0
            else:
                corr = np.corrcoef(f_vals, s_vals)[0, 1]
            
            
            if corr < 0:
                plot_values.append(-magnitude) 
                colors.append('#C84663')      
            else:
                plot_values.append(magnitude)  
                colors.append('#4E6A89')      
            
            
            simplified_name = feat_name.replace('reading', 'r').replace('monologue', 'm')
            plot_labels.append(simplified_name)

       
        plot_features = plot_labels[::-1]
        plot_values = plot_values[::-1]
        colors = colors[::-1]
        
        
        plt.figure(figsize=(12, 12)) 
        ax = plt.gca()
        
        
        bars = ax.barh(range(len(plot_features)), plot_values, height=0.7, align='center')
        
       
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
        
        
        ax.axvline(0, color='black', linewidth=1.2, zorder=3)
        
        
        ax.set_yticks(range(len(plot_features)))
        ax.set_yticklabels(plot_features, fontsize=11, fontweight='medium')
        
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        
        max_val = max(np.abs(plot_values)) * 1.15 if plot_values else 1.0
        ax.set_xlim(-0.3, 0.3)
        
        
        ylim = ax.get_ylim()
        ax.text(-max_val*0.5, ylim[0]-1, "High Value → MDDonly", color='#C84663', 
                ha='center', va='top', fontsize=12, fontweight='bold')
        ax.text(max_val*0.5, ylim[0]-1, "High Value → MDDRBD", color='#4E6A89', 
                ha='center', va='top', fontsize=12, fontweight='bold')

        plt.xlabel('SHAP Impact Magnitude & Direction', fontsize=12, fontweight='bold')
        plt.title(f"Feature Directionality (Sorted by Input) - {top_model_name}", fontsize=14, pad=15)
        
        plt.tight_layout()
        
        save_path_bar = output_dir / "plots" / "SHAP_BiDirectional_Sorted.png"
        plt.savefig(save_path_bar, dpi=300, bbox_inches='tight')
        plt.close()
        

    except Exception as e:
        print(f" [SHAP Error] {e}")
        import traceback
        traceback.print_exc()