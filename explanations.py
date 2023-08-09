"""
explanations.py
~~~~~~~~~~~

A module specific for explanation methods.



"""
import numpy as np
from secml.array import CArray
from secml.explanation import CExplainerIntegratedGradients
from sklearn import svm

class Explain():
    def __init__(self):
        self.model = None
        self.test_data_X = []
        self.test_data_y = []

    def set_input_data(self, model, test_data_X, test_data_y):
        self.model = model
        self.test_data_X = test_data_X
        self.test_data_y = test_data_y


    def IG(self):
        number_of_samples = len(self.test_data_y)
        integrated_grad_matrix = np.zeros((number_of_samples,len(self.test_data_X[0])))
        reference = CArray(np.zeros((1,len(self.test_data_X[0]))))

        for n in range(number_of_samples):
            X = CArray(np.array(self.test_data_X[n]), tosparse=False)
            attr_c = CExplainerIntegratedGradients(self.model).explain(X, y=int(self.test_data_y[n]), reference=reference)
            integrated_grad_matrix[n][:] = attr_c.get_data()

        return integrated_grad_matrix

        

    def SHAP(self):
        model = svm.SVC(C = 1.0, kernel='linear')
        model.fit(self.test_data_X,self.test_data_y)

        explainer = shap.KernelExplainer(model.predict, shap.sample(np.zeros((1000,len(self.test_data_X[0]))),10))
        result = explainer.shap_values(self.test_data_X)
        return result