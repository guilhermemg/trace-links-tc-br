import pandas as pd
from abc import ABCMeta, abstractmethod

class GenericModel(metaclass=ABCMeta):
    def __init__(self):
        self.name = None
        self.model_gen_name = None    
    
    def set_name(self, name):
        self.name = name
    
    def set_model_gen_name(self, gen_name):
        self.model_gen_name = gen_name
    
    @abstractmethod
    def recover_links(self, corpus, query, use_cases_names, bug_reports_names):
        pass
    
    def save_sim_matrix(self):
        self._sim_matrix.to_csv('models_sim_matrix/{}.csv'.format(self.get_model_gen_name()))
       
    def get_name(self):
        return self.name
        
    def get_sim_matrix(self):
        return self._sim_matrix
                                
    def get_model_gen_name(self):
        return self.model_gen_name
    
    @abstractmethod
    def model_setup(self):
        pass