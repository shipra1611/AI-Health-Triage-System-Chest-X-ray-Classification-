import torch
import torch.nn as nn
import torchvision.models as models

def create_untrained_model():
    print("Initializing DenseNet121 architecture...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    # Match the trained model's classification head
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 4)
    )
    
    print("Saving untrained weights to best_model.pth...")
    torch.save(model.state_dict(), 'best_model.pth')
    print("Saved successfully. The Streamlit app will now use real DenseNet inference (untrained).")

if __name__ == "__main__":
    create_untrained_model()
