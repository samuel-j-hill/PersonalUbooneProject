import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_msssim

class Config():
  def __init__(self):
    self.input_path = "combined_array_ds24.npy"
    self.output_directory = "ubooneoutput/"
    self.n_epochs = 1000
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

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

  with torch.no_grad():
    original_tensor = input_tensor[image_index].view(534, 1200).to(device)
    original_array = original_tensor.cpu().detach().numpy()
    original = ax1.imshow(original_array, cmap="jet")
    fig.colorbar(original, ax=ax1)
    ax1.set_title("Original")

    reconstructed_tensor = model(original_tensor.view(1, 1, 534, 1200)).to(device)
    reconstructed_tensor = reconstructed_tensor.reshape(534,1200)
    reconstructed_array = reconstructed_tensor.cpu().detach().numpy()
    reconstructed = ax2.imshow(reconstructed_array, cmap="jet")
    fig.colorbar(reconstructed, ax=ax2)
    ax2.set_title("Reconstructed")

  plt.savefig(config.output_directory + f"reconstruction_plot_{image_index+1}.png")
  plt.show()