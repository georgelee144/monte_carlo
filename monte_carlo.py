'''
This is going to be a libary for a monte carlo simulation. Mostly going to be used in finance setting
-didn't like the current offerings out there
'''

import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

import dash
import dash_core_components as dcc
import dash_html_components as html

class montecarlo():

    def __init__(self,returns_iter, starting_value, periods,sims=10000, action = "multiply", min_dd = None, max_dd = None, seed = None):

        self.returns_iter = returns_iter
        self.starting_value = starting_value
        self.periods = periods
        self.sims = sims
        self.action = action
        self.min_dd = min_dd
        self.max_dd = max_dd
        
        self.seed = seed
        
        if self.seed:
            np.random.seed = self.seed

        self.__start_df = {}        
        
        for simulation_requests in range(1,self.sims+1):
            self.__start_df[f'run_{simulation_requests}'] = list(self.starting_value)
        
        self.df = pd.DataFrame(data = self.starting_value) 

    def __run_monte(self):   
        
        for period in self.periods:
            self.df = pd.concat([self.df, self.get_next_row], ignore_index=True)

    def get_next_row(self):

        data_for_new_row = {}

        if self.action == 'add':

            for simulation_requests in range(1,self.sims+1):
                data_for_new_row[f'run_{simulation_requests}'] = self.df[f'run_{simulation_requests}'].iloc[-1] + self.__get_random_factor

                if self.min_dd or self.max_dd:
                    if self.__check_min_max(data_for_new_row[f'run_{simulation_requests}']):
                        data_for_new_row[f'run_{simulation_requests}'] = self.df[f'run_{simulation_requests}'].iloc[-1]


        if self.action == 'multiply':

            for simulation_requests in range(1,self.sims+1):
                data_for_new_row[f'run_{simulation_requests}'] = self.df[f'run_{simulation_requests}'].iloc[-1] * self.__get_random_factor

                if self.min_dd or self.max_dd:
                    if self.__check_min_max(data_for_new_row[f'run_{simulation_requests}']): 
                        data_for_new_row[f'run_{simulation_requests}'] = self.df[f'run_{simulation_requests}'].iloc[-1]


        return pd.DataFrame(data = data_for_new_row)

    
    def __get_random_factor(self):
        '''
        returns a randomly selected 

        may implement other features: no replacement
        '''

        return np.random.choice(self.returns_iter)

    def __check_min_max(self,data_for_new_row):

        if self.min_dd and data_for_new_row < self.min_dd:
            return False

        if self.max_dd and data_for_new_row > self.max_dd:
            return False

    def paramaters(self):

        paramaters_used = {'returns used' : self.returns_iter,  
                            'start' : self.starting_value, 
                            'number of simulations' : self.sims, 
                            'period per simulation' : self.periods ,
                            'function used (add or multiply)' : self.action, 
                            'max value allowed' : self.min_dd,
                            'min value allowed' : self.max_dd,
                            'seed':np.random.seed
        }

        for key,item in paramaters_used:
            print(f'{key}: {item}')

        return paramaters_used

    def summary(self):

        summary_stats = {'max' : self.df.max(),  
                         'min' : self.df.min(), 
                         'avg end' : self.df.mean(axis=0)[-1], 
                }

        for key,item in summary_stats:
            print(f'{key}: {item}')

        return summary_stats

    def plot(self,figsize = None):

        if figsize:
            plt.figure(figsize = figsize)

        plt.title(f'{self.sims}')
        plt.xlabel('Period')
        for column in self.df:
            plt.plot(self.df[column])

        plt.axhline(self.starting_value, color='black')
        plt.legend(loc = 'best')
        plt.show()

        


    def dash_plot(self):

        app = dash.Dash(__name__)

        app.layout = html.Div(

            dcc.Graph(
                id='montecarlo_graph',
                figure={
                    'data': [{'x' : self.df.index(),
                              'y' : self.df[f'{column}'],
                              'text' : f'{column}',
                              'name' : f'{column}'}] for column in self.df

                        }

            )
            )

        return app
        #if __name__ == '__main__':
        #    app.run_server(debug=True)
