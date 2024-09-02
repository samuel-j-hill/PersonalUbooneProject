import numpy as np
import matplotlib.pyplot as plt
import torch

class Config():
  def __init__(self):
    self.input_path = "combined_array_ds24.npy"
    self.output_directory = "ubooneoutput/"
    self.n_epochs = 10
    self.lr = 0.0001
    self.input_dim = 534*1200
    self.test_size = 0.2
    self.batch_size = 8
    self.compression_ratio = 500
    self.z_dim = int(np.ceil(self.input_dim/self.compression_ratio))
    self.num_workers = 2
    self.weighting_parameter = 50

        
  def save_config(self): 
    with open(self.output_directory +  "config.txt", 'w') as file:
        for attr in dir(self):
            if not (attr.startswith("__")):
                output_string = attr + " = " + str(getattr(self, attr))
                file.write(output_string)
                file.write("\n")
            
    return None
        

def save_output(config, model):

    torch.save(model.state_dict(), config.output_directory + "model.pt")
    config.save_config()

    return None

        
def weighted_mse(inputs, outputs, weighting_parameter):

  filter = torch.where(inputs>0.1, weighting_parameter, 1)
  reconstruction_error = outputs - inputs
  weighted_reconstruction_error = filter * reconstruction_error
  weighted_mse = torch.mean(weighted_reconstruction_error ** 2)

  return weighted_mse