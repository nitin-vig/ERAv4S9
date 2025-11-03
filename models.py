"""
Model architecture module with ResNet50 for ImageNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetImageNet(nn.Module):
    """ResNet tailored for ImageNet with proper initial layers"""
    
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetImageNet, self).__init__()
        self.in_channels = 64

        # Standard ImageNet initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50_imagenet(num_classes=1000, pretrained=False):
    """ResNet50 for ImageNet"""
    model = ResNetImageNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    
    # Load torchvision pretrained weights if requested
    if pretrained:
        try:
            import torchvision.models as tv_models
            pretrained_model = tv_models.resnet50(weights='IMAGENET1K_V2')
            
            # Copy all weights except the final fc layer
            model_dict = model.state_dict()
            pretrained_dict = pretrained_model.state_dict()
            
            # Filter out fc layer weights (different num_classes)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape 
                             and 'fc' not in k}
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"✅ Loaded ImageNet pretrained weights from torchvision")
        except Exception as e:
            print(f"⚠️  Could not load pretrained weights: {e}")
    
    return model

def get_model(model_name="resnet50", dataset_name="imagenette", num_classes=None, pretrained=False):
    """Get the appropriate model for the dataset - supports multiple ImageNet variants"""
    
    if num_classes is None:
        # Pass dataset_name to get the correct config for this stage
        dataset_config = Config.get_dataset_config(dataset_name)
        num_classes = dataset_config["classes"]
        print(f"📊 Creating model for {dataset_name} with {num_classes} classes")
    
    if dataset_name in ["imagenet", "imagenet1k", "imagenet_mini"]:
        if model_name == "resnet50":
            return resnet50_imagenet(num_classes=num_classes, pretrained=pretrained)
        else:
            raise ValueError(f"Model {model_name} not supported for {dataset_name}")
    
    elif dataset_name in ["tiny_imagenet", "imagenette"]:
        if model_name == "resnet50":
            return resnet50_imagenet(num_classes=num_classes, pretrained=pretrained)
        else:
            raise ValueError(f"Model {model_name} not supported for {dataset_name}")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: imagenet, imagenet1k, imagenet_mini, tiny_imagenet, imagenette")

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model, input_size=(3, 224, 224)):
    """Get model summary"""
    try:
        from torchsummary import summary
        return summary(model, input_size=input_size)
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        return None

def get_id_to_class_mapping(dataset_or_loader):
    """
    Extract id-to-class mapping from a dataset or DataLoader.
    
    Args:
        dataset_or_loader: Either a Dataset instance or DataLoader instance
        
    Returns:
        dict: Mapping from class_id (int) to class_name (str), or None if not available
    """
    # Get dataset from loader if needed
    dataset = dataset_or_loader
    if hasattr(dataset_or_loader, 'dataset'):
        dataset = dataset_or_loader.dataset
    
    if dataset is None:
        return None
    
    # Try to get class mapping
    id_to_class = None
    
    # Method 1: Check for class_to_idx (common in torchvision ImageFolder)
    if hasattr(dataset, 'class_to_idx') and isinstance(dataset.class_to_idx, dict):
        # Reverse the mapping: idx -> class_name
        id_to_class = {idx: class_name for class_name, idx in dataset.class_to_idx.items()}
    
    # Method 2: Check for classes attribute (list of class names)
    elif hasattr(dataset, 'classes') and isinstance(dataset.classes, (list, tuple)):
        id_to_class = {idx: str(class_name) for idx, class_name in enumerate(dataset.classes)}
    
    return id_to_class

def save_model(model, path, epoch=None, optimizer=None, scheduler=None, loss=None, id_to_class=None):
    """Save model checkpoint with required id-to-class mapping"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # id_to_class mapping is required
    if id_to_class is None:
        raise ValueError("id_to_class mapping is required when saving model. Provide dataset/loader or mapping explicitly.")
    
    checkpoint['id_to_class'] = id_to_class
    print(f"✅ Class mapping included in checkpoint ({len(id_to_class)} classes)")
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(model, path, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    
    print(f"Model loaded from {path}")
    return epoch, loss
