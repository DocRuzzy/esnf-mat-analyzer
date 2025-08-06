import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from esnf_mat_analyzer.data.synthetic_dataset import SyntheticRulerDataset
from esnf_mat_analyzer.processing.deep_gp_module import DeepGPModule
import torch.nn as nn

def train(model, dataloader, epochs=10):
    """
    A simple training loop for the DeepGPModule.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs = data['image']
            labels = data['gp_params']

            # To use the model, we need to convert the image to a 1D signal.
            # For now, we will just take the mean of the image across the color channels
            # and then flatten it. This is a placeholder and should be replaced with a more
            # sophisticated method.
            inputs = torch.mean(inputs.float(), dim=1).view(inputs.size(0), 1, -1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    print("Finished Training")

def main():
    """
    Main function to run the ablation study.
    """
    # Create the dataset and dataloader
    dataset = SyntheticRulerDataset(num_samples=100) # Small dataset for demonstration
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Create the model
    model = DeepGPModule()

    # Train the model
    train(model, dataloader)

    # In a real scenario, we would save the model, but we skip it here
    # to avoid issues with the sandbox environment.
    # torch.save(model.state_dict(), "deep_gp_model.pth")
    # print("Trained model saved to deep_gp_model.pth")
    print("Training finished. Model not saved.")

if __name__ == '__main__':
    main()
