import torch 
def save_full_model(model, file_path):
    torch.save(model, file_path)
    print(f"Full model saved to {file_path}")

# Loading the full model
def load_full_model(file_path, device='cpu'):
    model = torch.load(file_path, map_location=device)
    model.eval()  # Set to evaluation mode
    return model