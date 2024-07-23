import matplotlib.pyplot as plt

def visualize_comparison(input_image, output_image):
    # Assuming input_image and output_image are numpy arrays of the same dimension
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Plot input image
    ax = axes[0]
    ax.imshow(input_image, cmap='gray')
    ax.set_title('Input Image')
    ax.axis('off')  # Hide axes ticks

    # Plot output image
    ax = axes[1]
    ax.imshow(output_image, cmap='gray')
    ax.set_title('Output Image')
    ax.axis('off')  # Hide axes ticks

    plt.show()


import torch
from autoencoder_net import MyNet, MyDataset

# Define your model structure here

def main():

  device = torch.device("cpu")
  model = MyNet().to(device)
  #
  # Save weights
  # torch.save(model.state_dict(), 'model_weights.pth')

  # Later or elsewhere in your code
  # Ensure the model structure is the same as when the weights were saved
  model.load_state_dict(torch.load('mnist_cnn.pt'))
  model.eval()

  myDataset = MyDataset()
  dataset_test = myDataset.get('test')

  # Assuming `input_image` is your input tensor
  with torch.no_grad():
      for i in range(10):
        input_image, label = dataset_test[i]
        print(f"[DBG] input_image.shape: {input_image.shape}")
        unsqueezed_input_image = input_image.unsqueeze(0).to(device)
        # input_image = input_image.to(device)
        print(f"[DBG] unsqueezed_input_image.shape: {input_image.shape}")
        output_image = model(unsqueezed_input_image)
        print(f"[DBG] output_image.shape: {output_image.shape}")
        output_image = output_image.squeeze(0)
        print(f"[DBG] output_image.shape: {output_image.shape}")
        visualize_comparison(input_image.squeeze(), output_image.squeeze())
        # Post-process and visualize `output_image` as needed

if __name__ == "__main__":
    main()
