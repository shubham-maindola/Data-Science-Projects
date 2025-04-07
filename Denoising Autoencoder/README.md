## **Denoising Autoencoder for Image Restoration**  

This project implements a **denoising autoencoder** trained on a subset of **100K ImageNet images** to remove noise and blur from images. The model is built using **PyTorch** and learns to restore noisy, slightly blurry images to their original clean versions.  

### **Features**  
- **Autoencoder Architecture**: Uses convolutional layers for encoding and decoding.  
- **Custom Loss Function**: Combines **L1 loss, perceptual loss (VGG16), and SSIM loss** for better restoration quality.  
- **Realistic Noisy Data**: Training images were corrupted with **Gaussian blur and salt & pepper noise**.  
- **Trained on Large Dataset**: 100K images from **ImageNet** used for training.  

### **Usage**  
1. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
2. **Load the trained model**:  
   ```python
   import torch
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   model = torch.load("denoising_autoencoder_model.pth")
   model.to(device)
   model.eval()
   ```
3. **Test on a new image**:  
   ```python
   import numpy as np
   import torchvision.transforms as transforms
   from PIL import Image
   import matplotlib.pyplot as plt

   def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
       noisy_image = np.copy(image)
    
       num_salt = np.ceil(salt_prob * image.size)
       salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
       noisy_image[salt_coords[0], salt_coords[1]] = 255

       num_pepper = np.ceil(pepper_prob * image.size)
       pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
       noisy_image[pepper_coords[0], pepper_coords[1]] = 0

       return noisy_image

   input_transform = transforms.Compose([
       transforms.Resize((256, 256)), 
       transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)), 
       transforms.Lambda(lambda img: add_salt_pepper_noise(np.array(img), salt_prob=0.005, pepper_prob=0.005)), 
       ])

   with torch.no_grad():
        clean_image = Image.open(file_path).convert("RGB")
        input_image = (torch.from_numpy(input_transform(clean_image)).permute(2, 0, 1).float() / 255.0).to(device)
        output = model(input_image)
        
   denoised_image = output.squeeze(0).cpu().detach() 
   denoised_image = transforms.ToPILImage()(denoised_image)

   fig, ax = plt.subplots(1, 2, figsize=(10, 4))
   ax[0].imshow(input_image.cpu().permute(1,2,0))
   ax[0].set_title("Input Image")
   ax[1].imshow(denoised_image)
   ax[1].set_title("Denoised Output")
   plt.show()
   ```

### **Training Details**  
- **Dataset**: 100K ImageNet subset  
- **Input**: Noisy, blurry images (256x256)  
- **Output**: Clean, original images (256x256)  
- **Optimizer**: Adam  
- **Loss Function**: L1 loss + 0.1 * Perceptual loss (VGG16) + 0.5 * SSIM loss  

### **Results**  
The model effectively removes noise and blur while preserving fine details, leveraging a combination of **pixel-wise and perceptual loss functions**.  
