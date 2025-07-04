from lime.lime_text import LimeTextExplainer
from utils.predictor import predict_proba
import numpy as np

class_names = ['Fake', 'Real']
explainer = LimeTextExplainer(class_names=class_names)

def lime_explanation(text):
    def predictor_fn(texts):
        return np.array([predict_proba(t) for t in texts])

    exp = explainer.explain_instance(text, predictor_fn, num_features=10)
    return exp.as_html()