import torch
import numpy as np
from lsdo_airfoil.core.models.scalar_valued_regressions.scalar_valued_neural_nets import scaler_valued_nn_model, scaler_valued_nn_model_gelu
from lsdo_airfoil.core.models.vector_valued_regressions.vector_valued_neural_nets import LSTM
from lsdo_airfoil import MODELS_FOLDER
import copy


neural_net_model_dict = dict()
input_dim = 35
hidden_dim_rnn = 128
output_dim_rnn = 100
num_layers_rnn = 1

def get_airfoil_models():    
    scalar_valued_models = ['Cl', 'Cd', 'Cm']
    vector_valued_models = ['Cp', 'Ue']
    
    for model in scalar_valued_models:
        neural_net_model = scaler_valued_nn_model
        neural_net_model.load_state_dict(torch.load(MODELS_FOLDER / f'scalar_valued_regressions/{model}_model'))
        neural_net_model.eval()
        neural_net_model.requires_grad_(False)
        neural_net_model_dict[model] = copy.deepcopy(neural_net_model)
    

    for model in vector_valued_models:
        neural_net_upper_model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim_rnn, output_dim=output_dim_rnn, num_layers=num_layers_rnn)
        neural_net_lower_model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim_rnn, output_dim=output_dim_rnn, num_layers=num_layers_rnn)
        neural_net_upper_model.eval()
        neural_net_lower_model.eval()
        neural_net_upper_model.requires_grad_(False)
        neural_net_lower_model.requires_grad_(False)

        neural_net_upper_model.load_state_dict(torch.load(MODELS_FOLDER / f'vector_valued_regressions/{model}_upper_model'))
        neural_net_lower_model.load_state_dict(torch.load(MODELS_FOLDER / f'vector_valued_regressions/{model}_lower_model'))

        neural_net_model_dict[f"{model}_upper"] = copy.deepcopy(neural_net_upper_model)
        neural_net_model_dict[f"{model}_lower"] = copy.deepcopy(neural_net_lower_model)
        
    return neural_net_model_dict

