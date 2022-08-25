---
layout: post
title: "Who's That Pokémon?"
subtitle: "A Generative Adversarial Network (GAN) that produces never before seen Pokémon"
# date: 2022-07-15 23:45:13 -0400
background: '/img/posts/1.png'
---

<h2 class="section-heading">Introduction</h2>>

<p>Generative Adversarial Networks (GANs) are a type of unsupervised neural networks that falls under the purview of deep learning models. They are commonly used in the image-processing domain to create art <b>[1]</b>, music <b>[2]</b>, or to improve the quality of low-resolution images/videos <b>[3]</b>. Recently, researchers at the University of Toronto used their applications in biochemistry and medical studies to generate 30,000 designs for six different new compounds that were found to imitate drug-like properties and target a protein involved in fibrosis <b>[4]</b>. I trained a GAN model to generate fake Pokémon.</p>

<p>Because GANs are primarily taught to learn the distribution of any given dataset, the applications are really domain-independent. GANs will be able to replicate aspects of our environment given a well-defined dataset. The key constraint is the computing power required to train these models, which is further hampered by the fact that they are notoriously difficult to train, necessitating extra training time and computational power.</p>

<p>What makes training these models so difficult? Understanding this requires taking a look under the hood of this model. It was first proposed in the landmark paper Generative Adversarial Nets <b>[5]</b>, and it presents a paradigm in which two fully-connected neural networks (NN) compete in a zero-sum game. One of the NNs, known as the generative network or the generator, will work to generate "false" data, while the other, known as the discriminative network or the discriminator, will work to evaluate and distinguish between the actual and fake data.</p>

![Figure 1](/img/posts/fig1.png)

<p>Figure 1 depicts an abstract representation of how the generator (green line) would "dupe" the discriminator. The generator will train until its distribution resembles that of the real dataset (black dotted line). Given that the generator's job is to trick the discriminator until it can no longer distinguish between the two distributions, the discriminative distribution (blue dashed line) should flatten when the fake and actual distributions become indistinguishable.</p>

<p>The model used to create the "fake" Pokémon in this case is known as the DCGAN, which stands for Deep Convolutional Generative Adversarial Network. This model, unlike the fully connected models suggested in <b>[5]</b> employs two convolutional neural networks (CNNs) for the generative and discriminative networks.
The discriminator is a CNN model, whereas the generator is a deconvolutional neural network, which works inversely to a conventional CNN model. Where a CNN learns the spatial hierarchies of features within an image, moving from granular to high level details, the deconvolutional neural network or the generator learns to convert the latent space inputs into an actual image, generating meaning from noise, by regularly updating its weights by learning how the discriminator evaluates the images fed into its network. This is depicted in the figure below, which shows how data flows through a generative neural network.</p>

![Figure 2](\img\posts\fig2.png)

<p>By providing a random seed, the generator begins to produce candidates for the discriminator from a latent space and maps it to the distribution of the dataset being used. A latent space is a representation of compressed data best explained in <b>[7]</b>. The space is initially populated randomly, but as the generator begins to understand a dataset’s distribution, the latent space would slowly start to be populated by features learned from the distribution. In contrast, the discriminator is trained on random datapoints drawn from the actual dataset. Both models will be trained until they achieve an acceptable level of accuracy, with each model undergoing backpropagation individually to enhance accuracy.</p>

<p>This is further emphasized in Figure 3, where we see how the data produced by the generator is fed to the discriminator along with the real data.</p>

![Figure 3](\img\posts\fig3.png)

<h2 class="section-heading">Data Collection and Processing</h2>

<p>For this project, the Pokémon dataset was acquired via Kaggle. The original dataset is made up of 819 photos that were uploaded as .png files with a resolution of 256x256 pixels <b>[9]</b>. Because GANs are notoriously data hungry <b>[10]</b>, the size of this dataset was expanded 13 times prior to training by executing a data augmentation step.</p>

![Figure 4](\img\posts\fig4.png)

<p>Despite expanding the dataset size, the results appeared to follow the same patterns as those seen  on Kaggle <b>[9]</b>. Normalizing the dataset by calculating the mean and standard deviation did not appear to improve the results and actually worsened them.</p>

<p>A few considerations regarding the hyperparameters were made before training the model. The training was repeated four times to see if altering certain hyperparameters affected the quality of the results. The <b>learning rate</b>, <b>batch size</b>, <b>latent space</b>, and <b>number of epochs</b> were all altered. Because of memory constraints caused by the number of layers in the models, the 256x256 input picture was scaled down to accept 64x64 and 128x128 images. The change in the input did not appear to drastically change the resolution of the outputs either.</p>

<p>Setting the batch size to a smaller value would prevent the discriminator from quickly outperforming the generator, leading to poorer results. The learning rate was also set to a conservative value; 1e-4 as it led to better results, purely based on the results observed over the 4 iterations. The latent space was primarily changed based on the assumption that because this value represented “compressed” data, the generator would reconstruct the distribution of the dataset from a larger value (or from larger latent space), ultimately leading to better results. Finally, the training time or epochs were chosen based on prior implementations of this project, other similar projects, and the constraints of my PC.</p>

<p>The values that were selected for each of the iterations are presented in the <b>Table 1</b> below</p>  


{:.mbtablestyle}
<table style="margin-left:auto;margin-right:auto;">
<colgroup>
<col width="15%" />
<col width="25%" />
<col width="20%" />
<col width="25%" />
<col width="30%" />
<col width="30%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align:center">Run</th>
<th style="text-align:center">Model Version</th>
<th style="text-align:center">Batch Size</th>
<th style="text-align:center">Learning Rate</th>
<th style="text-align:center">Latent Space</th>
<th style="text-align:center">Epochs</th>
</tr>
</thead>
<tbody>
<tr>
<td markdown="span" style="text-align:center">1</td>
<td markdown="span" style="text-align:center">64 px</td>
<td markdown="span" style="text-align:center">128</td>
<td markdown="span" style="text-align:center">1.00E-04</td>
<td markdown="span" style="text-align:center">64</td>
<td markdown="span" style="text-align:center">100</td>
</tr>
<tr>
<td markdown="span" style="text-align:center">2</td>
<td markdown="span" style="text-align:center">64 px</td>
<td markdown="span" style="text-align:center">64</td>
<td markdown="span" style="text-align:center">2.00E-04</td>
<td markdown="span" style="text-align:center">128</td>
<td markdown="span" style="text-align:center">70</td>
</tr>
<tr>
<td markdown="span" style="text-align:center">3</td>
<td markdown="span" style="text-align:center">128 px</td>
<td markdown="span" style="text-align:center">128</td>
<td markdown="span" style="text-align:center">2.00E-04</td>
<td markdown="span" style="text-align:center">256</td>
<td markdown="span" style="text-align:center">200</td>
</tr>
<tr>
<td markdown="span" style="text-align:center">4</td>
<td markdown="span" style="text-align:center">128 px</td>
<td markdown="span" style="text-align:center">64</td>
<td markdown="span" style="text-align:center">1.00E-04</td>
<td markdown="span" style="text-align:center">256</td>
<td markdown="span" style="text-align:center">100</td>
</tr>
</tbody>
</table>
{:class="table table-bordered"}

<h2 class="section-heading">Results</h2>

<p>The results for each of the runs, presented in <b>Table 1</b> are presented below</p>

<h3><i>RUN 1</i></h3>

![Figure 5](\img\posts\fig5.gif)<br>
<p>
<br>
</p>

![Figure 6](\img\posts\fig6.png)

<p>The Pokémon generated using this model have distinct shapes and colours, but they lack features such as faces, limbs, or appendages such as tails, wings, horns, fins, and so on that are commonly seen on Pokémon. The losses for both models appear to raise concerns about mode collapse and/or failure of convergence based on the loss plot. When the generator's loss begins to oscillate repeatedly with the same <b><a href="https://machinelearningmastery.com/wp-content/uploads/2019/07/Line-Plots-of-Loss-and-Accuracy-for-a-Generative-Adversarial-Network-with-Mode-Collapse.png">oscillation loss pattern</a></b>, mode collapse might have occurred. It also results in very little diversity among the samples generated. However, the outcomes are far from identical. While it is evident that the loss functions for the generator and discriminator do not converge, it would also lead to the results simply producing plain noise as in Figure 7 below.</p>

![Figure 7](\img\posts\fig7.gif)

<p>As a result, in addition to determining the best combination of hyperparameters, three additional runs were carried out to see whether similar patterns in the loss functions from Figure 6 maintained.</p>

<h3><i>RUN 2</i></h3>

![Figure 8](\img\posts\fig8.gif)<br>
<p>
<br>
</p>

![Figure 9](\img\posts\fig9.png)

<h3><i>RUN 3</i></h3>

![Figure 10](\img\posts\fig10.gif)<br>
<p>
<br>
</p>

![Figure 11](\img\posts\fig11.png)<br>
<p>
<br>
</p>

![Figure 12](\img\posts\fig12.png)

<h3><i>RUN 4</i></h3>

![Figure 13](\img\posts\fig13.gif)<br>
<p>
<br>
</p>

![Figure 14](\img\posts\fig14.png)<br>
<p>
<br>
</p>

![Figure 15](\img\posts\fig15.png)<br>
<p>
<br>
</p>

![Figure 16](\img\posts\fig16.png)

<p>The loss functions for the generator and discriminator in Figures 9, 10, and 12 are observed to follow a general trend, seen in Figure 6. The results obtained throughout each of these runs are also identical to the ones shown in Figure 5. However, when examined through a forced creative lens, the results of the third run appear to show some form of limbs and appendages. Except for a couple in the last row of Figure 11, none of the results are truly legible. Because this model was trained for the longest time, 200 epochs, there is a significant likelihood that training the DCGAN model using the hyperparameters from RUN 3 for an even longer time will result in more defined outcomes.</p>

<p>Despite the slightly underwhelming results, I believe the project has great promise. Curating a more well-defined dataset with greater care to the data augmentation step could yield more clarity to the results presented. Furthermore, by labelling the data, and introducing noisy labels through the flipping the labels (by labelling real data as fake), the discriminator could be confused., thereby improving the results. In addition, I did not include many filters in the CNN models. Increasing the number of filters would also aid the CNN in extracting more information from images.</p>

<h2 class="section-heading">Code</h2>

<p>Lastly, the code used in this project is presented below:</p>

<p><i><b>Importing all relevant modules for loading 
and transforming the data for the model</b></i></p>

```python
import os 
import torch
import pandas as pd
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
```

<p><i><b>Created a class to load the data in
and defined a few helper functions.</b></i></p>

```python
class PokeData(Dataset):
    # Initializing the variables  
    def __init__(self, csv_file, root_dir, transform=None):
        self.filenames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):      
        # returns the size of the dataset
        return len(self.filenames)
    
    def __getitem__(self, index): 
        # loads in the images by joining the directory with the file names
        # in the dataframe
        image_path = os.path.join(self.root_dir, self.filenames.iloc[index, 0])
        img = io.imread(image_path)

        if self.transform:
            img = self.transform(img)
        
        return img
    
    # function to check if the dataset has loaded in properly
    def show_img(dataset):
        plt.figure()
        for i in range(len(dataset)):
            sample = dataset[i]

            ax = plt.subplot(2, 2, i +1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            plt.imshow(sample)

            if i == 3:
                plt.show()
                break
    
    # calculates the mean and standard deviation 
    # for the dataset. Use with caution
    def get_mean_std(dataloader):
    # VAR[X] = E[X**2] - E[X]**2
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data in dataloader:
            channels_sum += torch.mean(data, dim=[0,2,3])
            channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
            num_batches += 1
    
        mean = channels_sum/num_batches
        std = (channels_squared_sum/num_batches - mean**2)**0.5

        return mean, std
    
    # creates a new folder and stores all the 
    # augmented photos in that folder
    def data_augmentation(dataset, augmented_dir, val=5): 
        if not os.path.exists(augmented_dir):
            os.mkdir(augmented_dir)
            img_num = 820
            for _ in range(val):
                for img in tqdm(dataset):
                    save_image(img, augmented_dir+"/"+str(img_num)+'.png')
                    img_num += 1 
```

<p><i><b>Selected a few filters<br>  
to apply on the dataset.</b></i></p>

```python
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.5, hue=0.5),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomSolarize(threshold=64, p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor()
    ])
```
<p><i><b>Loaded the data; the</b></i> <code>show_images</code> <i><b>variable is
used to check if the data was loaded in properly.</b></i></p>

```python
show_images = PokeData(csv_file=csv_file,
                            root_dir=root_dir)

pokemon_dataset = PokeData(csv_file=csv_file,
                            root_dir=aug_dir,
                            transform=my_transforms)

pokemon_dataloader = DataLoader(pokemon_dataset,
                                batch_size = 64, shuffle=True)


show_imgs = PokeData.show_img(show_images)

augmented_data_batchs = "archive/aug_batches"
aug_data = PokeData.data_augmentation(pokemon_dataset, aug_dir, 13)
print(aug_data)
```

<p>For this project, two model architectures were used; one that accepted 64x64 px images
and one that accepted 128x128 px images. The discriminator and generators for these two models
are presented below</p>

<p><i><b>pokeGAN_64x64 Discriminator</b></i></p>

```python
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ), # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), # 16 x 16
            self._block(features_d*2, features_d*4, 4, 2, 1), # 8 x 8 
            self._block(features_d*4, features_d*8, 4, 2, 1), # 4 x 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), #1x1
            nn.Sigmoid()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)
```

<p><i><b>pokeGAN_64x64 Generator</b></i></p>

```python
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0),
            self._block(features_g*16, features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g*2, channels_img, kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(), #Will produce an output between [-1, 1]
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential( 
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.gen(x)
```

<p><i><b>pokeGAN_128x128 Discriminator</b></i></p>

```python
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 128 x 128
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ), # 64 x 64
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), # 32 x 32
            self._block(features_d*2, features_d*4, 4, 2, 1), # 16 x 16 
            self._block(features_d*4, features_d*8, 4, 2, 1), # 8 x 8
            self._block(features_d*8, features_d*16, 4, 2, 1), # 4 x 4
            nn.Conv2d(features_d*16, 1, kernel_size=4, stride=2, padding=0), #1x1
            nn.Sigmoid()
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)
```
<p><i><b>pokeGAN_128x128 Generator</b></i></p>

```python
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0),
            self._block(features_g*16, features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),
            self._block(features_g*2, features_g, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g, channels_img, kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(), #Will produce an output between [-1, 1]
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential( 
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.gen(x)
```

<p><i><b>A function to initialize the weights of the models.</b></i></p>

```python
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
```

<p><i><b>And a test function to check if the models are working properly.</b></i></p>

```python
def test():
    N, in_channels, H, W = 8, 3, H, W 
    z_dim = 128
    x = torch.randn(N, in_channels, H, W)
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success")


test()
```