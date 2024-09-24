import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_msssim
from scipy.sparse import csr_matrix

class Config():
  def __init__(self):
    self.input_path = "combined_array_ds24.npy"
    self.output_directory = "ubooneoutput/"
    self.n_epochs = 5
    self.lr = 0.0001
    self.input_dim = 534*1200
    self.test_size = 0.2
    self.batch_size = 8
    self.compression_ratio = 10000
    self.z_dim = int(np.ceil(self.input_dim/self.compression_ratio))
    self.num_workers = 2
    self.weighting_parameter = 50
    self.L1 = False
    self.L1_weight = 1e-7
    self.SSIM = False
    self.Sparse = True

        
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
  
  
def train(model, optimizer, device, train_dl, test_dl, config):
  
  train_loss_values = []
  test_loss_values = []

  for epoch in range(config.n_epochs):

      print(f"Epoch {epoch + 1}")
      running_train_loss = 0
      running_test_loss = 0

      model.train()
      for train_idx, (inputs,) in enumerate(train_dl):

          inputs = inputs.view(-1, 1, 534, 1200).requires_grad_().to(device)

          optimizer.zero_grad()

          outputs = model(inputs)

          if config.L1:
            loss = weighted_mse_L1(inputs, outputs, config.weighting_parameter, config.L1_weight, model)
          elif config.SSIM:
            loss, ssim_loss = weighted_mse_ssim(inputs, outputs, config.weighting_parameter)
          else:
            loss = weighted_mse(inputs, outputs, config.weighting_parameter)
          loss.backward()
          
          running_train_loss += loss.item()
          optimizer.step()

      model.eval()
      with torch.no_grad():

          for test_idx, (inputs,) in enumerate(test_dl):

              inputs = inputs.view(-1, 1, 534, 1200).to(device)

              outputs = model(inputs)

              if config.L1:
                loss = weighted_mse_L1(inputs, outputs, config.weighting_parameter, config.L1_weight, model)
              elif config.SSIM:
                loss, ssim_loss = weighted_mse_ssim(inputs, outputs, config.weighting_parameter)
              else:
                loss = weighted_mse(inputs, outputs, config.weighting_parameter)
              
              running_test_loss += loss.item()

      train_loss_values.append(running_train_loss / train_idx)
      test_loss_values.append(running_test_loss / test_idx)
      print("Train loss: ", running_train_loss)
      
  return model, train_loss_values, test_loss_values


def train_sparse(model, optimizer, device, train_dl, test_dl, config):
  
  train_loss_values = []
  test_loss_values = []

  for epoch in range(config.n_epochs):

      print(f"Epoch {epoch + 1}")
      running_train_loss = 0
      running_test_loss = 0

      model.train()
      for train_idx, (inputs,) in enumerate(train_dl):

          coords, features = dense_to_sparse(inputs)
          coords = coords.to(device)
          features = features.to(device)
          inputs = inputs.to(device)

          optimizer.zero_grad()

          outputs = model((coords, features))

          if config.L1:
            loss = weighted_mse_L1(inputs, outputs, config.weighting_parameter, config.L1_weight, model)
          elif config.SSIM:
            loss, ssim_loss = weighted_mse_ssim(inputs, outputs, config.weighting_parameter)
          else:
            loss = weighted_mse(inputs, outputs, config.weighting_parameter)
          loss.backward()
          
          running_train_loss += loss.item()
          optimizer.step()

      model.eval()
      with torch.no_grad():

          for test_idx, (inputs,) in enumerate(test_dl):

              coords, features = dense_to_sparse(inputs)
              coords = coords.to(device)
              features = features.to(device)
              inputs = inputs.to(device)

              outputs = model((coords, features))

              if config.L1:
                loss = weighted_mse_L1(inputs, outputs, config.weighting_parameter, config.L1_weight, model)
              elif config.SSIM:
                loss, ssim_loss = weighted_mse_ssim(inputs, outputs, config.weighting_parameter)
              else:
                loss = weighted_mse(inputs, outputs, config.weighting_parameter)
              
              running_test_loss += loss.item()

      train_loss_values.append(running_train_loss / train_idx)
      test_loss_values.append(running_test_loss / test_idx)
      print("Train loss: ", running_train_loss)
      
  return model, train_loss_values, test_loss_values


        
def weighted_mse(inputs, outputs, weighting_parameter):

  filter = torch.where(inputs>0.1, weighting_parameter, 1)
  reconstruction_error = outputs - inputs
  weighted_reconstruction_error = filter * reconstruction_error
  weighted_mse = torch.mean(weighted_reconstruction_error ** 2)

  return weighted_mse


def weighted_mse_L1(inputs, outputs, weighting_parameter_1, weighting_parameter_2, model):

  filter = torch.where(inputs>0.1, weighting_parameter_1, 1)
  reconstruction_error = outputs - inputs
  weighted_reconstruction_error = filter * reconstruction_error
  weighted_mse = torch.mean(weighted_reconstruction_error ** 2)

  l1_loss = 0
  for w in model.parameters():
    l1_loss += w.norm(1)

  return weighted_mse + weighting_parameter_2 * l1_loss


def weighted_mse_ssim(inputs, outputs, weighting_parameter):

  filter = torch.where(inputs>0.1, weighting_parameter, 1)
  reconstruction_error = outputs - inputs
  weighted_reconstruction_error = filter * reconstruction_error
  weighted_mse = torch.mean(weighted_reconstruction_error ** 2)

  ssim_loss = 1 - pytorch_msssim.ssim(inputs, outputs)

  return weighted_mse + (10 * ssim_loss)


def reconstruction_plotter(image_index, input_tensor, model, device, config):

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

  with torch.no_grad():
    original_tensor = input_tensor[image_index].view(534, 1200).to(device)
    original_array = original_tensor.cpu().detach().numpy()
    original = ax1.imshow(original_array, cmap="jet")
    fig.colorbar(original, ax=ax1)
    ax1.set_title("Original")
    
    if config.Sparse:
      original_tensor_sparse = dense_to_sparse(original_tensor.view(1, 534, 1200))
      reconstructed_tensor = model(original_tensor_sparse).to(device)
      reconstructed_tensor = reconstructed_tensor.reshape(534,1200)

    else:
      reconstructed_tensor = model(original_tensor.view(1, 1, 534, 1200)).to(device)
      reconstructed_tensor = reconstructed_tensor.reshape(534,1200)
      
    reconstructed_array = reconstructed_tensor.cpu().detach().numpy()
    reconstructed = ax2.imshow(reconstructed_array, cmap="jet")
    fig.colorbar(reconstructed, ax=ax2)
    ax2.set_title("Reconstructed")

    difference_tensor = original_tensor - reconstructed_tensor
    difference_array = difference_tensor.cpu().detach().numpy()
    difference = ax3.imshow(difference_array, cmap="jet")
    fig.colorbar(difference, ax=ax3)
    ax3.set_title("Difference")

  plt.savefig(config.output_directory + f"reconstruction_plot_{image_index+1}.png")
  plt.show()

  return None
  
  
def loss_plotter(train_loss_values, test_loss_values, config):

  plt.plot(range(len(train_loss_values)), train_loss_values, label="Training loss")
  plt.plot(range(len(test_loss_values)), test_loss_values, label="Test loss")
  plt.title(f"lr = {config.lr}, bs = {config.batch_size}, num_epochs = {config.n_epochs}")
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.legend()
  plt.savefig(config.output_directory + "loss_plot.png")
  plt.show()

  print(f"Final training loss: {train_loss_values[-1]}")
  print(f"Final test loss: {test_loss_values[-1]}")

  return None


def dense_to_sparse(input_tensor):
    
  input_array = input_tensor.cpu().numpy()
  
  coords = []
  features = []
  
  for idx, sub_array in enumerate(input_array):
    csr_mat = csr_matrix(sub_array)
    row_indices, col_indices = csr_mat.nonzero()
    
    batch_coords = np.stack([row_indices, col_indices, np.full(row_indices.shape,idx)], axis=1)
    batch_features = np.vstack(csr_mat.data)
    
    coords.append(batch_coords)
    features.append(batch_features)

  coords = np.concatenate(coords, axis=0)
  features = np.concatenate(features, axis=0)
  coords = torch.LongTensor(coords)
  features = torch.FloatTensor(features)
  
  return coords, features


def zero_suppression(input_array, cutoff):
    
    for sub_array in input_array:
        sub_array[sub_array < cutoff] = 0
        
    return input_array