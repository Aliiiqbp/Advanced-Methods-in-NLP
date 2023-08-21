import torch
import math
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Basic VAE with convolutional encoder and deconvolutional decoder.
    """
    def __init__(self, z_dim):
        """
        Initializes the layers of the VAE, which should include:
        - 1. one dropout layer
        - 2. a stack of convolutional layers (we recommend starting
          with 3 of them) interleaved with max pooling layers
        - 3. a dense layer to project the output from the final
          convolution down to size self.z_dim
        - 4. a dense layer to project the encoder output onto mu
        - 5. a dense layer to project the encoder output onto sigma
        - 6. a stack of deconvolutional layers (AKA transposed convolutional
          layers; we recommend starting with 4 of them) interleaved with
          2d batch normalization layers.

        Input:
        - z_dim:    size of the codes produced by this encoder
        """
        super(VAE, self).__init__()
        self.z_dim = z_dim

        # TODO 1.Dropout Layer
        self.dropout = nn.Dropout(0.5)
        
        # TODO 2.Encoder Layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )

        # TODO 3. Dense Layers to project the output from the final convolution down to size self.z_dim
        self.fc = nn.Linear(128*8*8, self.z_dim)

        # TODO 4. Dense layer to project the encoder output onto mu
        self.fc_mu = nn.Linear(self.z_dim, self.z_dim)

        # TODO 5. Dense layer to project the encoder output onto sigma
        self.fc_sigma = nn.Linear(self.z_dim, self.z_dim)

        # TODO 6. Decoder Layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Adjust the kernel size and padding here to get [50, 1, 64, 64]
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encode(x)

        mu_sigmas = self.project(z)
        mu, sigma = mu_sigmas


        z = self.reparametrize(mu, sigma)
        z = z.unsqueeze(-1).unsqueeze(-1)
        gen_images = self.decode(z)
        return gen_images, mu_sigmas
    

    def encode(self,x):
        """
        Given a sequence of images, applies a stack of convolutional layers to
        each image, flattens the convolved outputs, and projects each one to
        size self.z_dim.

        Input:
        - x:    torch.Tensor of shape (seq_length, n_channels, img_width, img_height)
                which equals (50, 1, 64, 64) with the default model configuration.

        Returns:
        - A torch.Tensor of size (seq_length, self.z_dim) which defaults to (50, 16)
        """
        #seq_length = x.shape[0]
        #n_channels = x.shape[1]

        # update the in_channels in the 1st lay of encoder 
        #self.encoder[0] = nn.Conv2d(in_channels=n_channels, out_channels = 16, kernel_size=3, stride = 1, padding = 1)
        
        #y = self.encoder(x)
        #y = y.view(seq_length, -1)

        #placeholder = torch.rand(seq_length, self.z_dim, requires_grad = True).to(x)
        #assert placeholder.size() == y.size(), "encoder error: size not match"


        y  = self.encoder(x)

        y = y.view(50, -1)

        y  = self.fc(y)

        return y


    def project(self, x):
        """
        Given an intermediate sequence of encoded images, applies two
        projections to each encoding to produce vectors mu and sigma.

        Input:
        - x:    torch.Tensor of shape (seq_length, self.z_dim) (output
                from self.encode)

        Returns:
        - A tuple of two torch.Tensors, each of shape (seq_length, self.z_dim)
        """
        seq_length = x.shape[0]
        
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        assert x.size() == mu.size(), "project error: mean size not match"
        assert x.size() == sigma.size(), "project error: val size not match"

        return (mu, sigma) 
    
    def reparametrize(self, mu, sigma):
        """
        Applies the reparametrization trick from
        https://arxiv.org/pdf/1312.6114v10.pdf

        Input:
        - mu:       torch.Tensor of shape (seq_length, self.z_dim) returned
                    by self.project()
        - sigma:    torch.Tensor of shape (seq_length, self.z_dim) returned
                    by self.project()

        Returns:
        - A sequence of codes z of shape (seq_length, self.z_dim) obtained by
          sampling from a normal distribution parameterized by mu and sigma
        """
        epislon = torch.randn_like(sigma)

        z = mu + sigma * epislon

        placeholder = mu + sigma

        assert placeholder.size() == z.size(), "reparameter error: size not match"

        return z

    def decode(self, z):
        """
        Given a sequence of variational codes, applies a stack of deconvolutional
        layers (AKA transposed convolutional layers) to recover a sequence of images.

        Input:
        - z:    torch.Tensor of shape (seq_length, self.z_dim) returned by
                self.reparametrize()

        Returns:
        - A sequence of images of shape (seq_length, n_channels, img_width, img_height)
          which defaults to (50, 1, 64, 64). All outputs should be in the range [0, 1].
        """


        seq_length = z.shape[0]
        assert seq_length == 50, "decode error: size of seq_length not match to default"
        n_channels = 1
        img_width = 64
        img_height = 64     

       # self.decoder[-2] = nn.ConvTranspose2d(in_channels=64, out_channels=n_channels, kernel_size=4, stride=2, padding=1)
        
        z = z.unsqueeze(-1).unsqueeze(-1)

        y = self.decoder(z)


        #placeholder = z.repeat_interleave(4, dim=1).unsqueeze(1).repeat(1,64,1).unsqueeze(1)
        #assert placeholder.size() == y.size(), "decode error: size not match"

        return y

def kld(mu, log_var):
    """
    Computes KL div loss wrt. a standard normal prior.

    Input:
    - log_var:  log variance of encoder outputs
    - mu:       mean of encoder outputs

    Returns:    D_{KL}(\mathcal{N}(mu, sigma) || \mathcal{N}(0, 1))
                = log(1 / sigma) + (sigma^2 + mu^2)/2 - 1/2
                = -0.5*(1 + log(sigma^2) - sigma^2 - mu^2)
    """
    #log_var = torch.clamp(log_var, min = -0.3, max = 0.3)
    #kld_loss = -0.5 * torch.sum( 1+log_var -log_var.exp()- mu.pow(2))
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kld_loss

def vae_loss(gen_images, input_images, mu_sigmas):
    """
    Computes BCE reconstruction loss and KL div. loss for VAE outputs.

    Input:
    - gen_images:   list of 2 decoded image sequences, each of shape (seq_length,
                    n_channels, img_width, img_height) which defaults to
                    (50, 1, 64, 64). In the baseline model, this will contain
                    one sequence decoded from the VAE itself, and another decoded
                    from the top layer of the Transformer.
    - input_images: list of target image sequences, each of shape (seq_length,
                    n_channels, img_width, img_height) which defaults to
                    (50, 1, 64, 64). The nth sequence in gen_images will be
                    evaluated against the nth sequence in input_images to
                    compute the reconstruction loss. In the baseline mode, this
                    will contain one sequence so is purly a tensor.
    - mu_sigmas:    list of (mu, sigma) tuples, where each mu and sigma is a
                    sequence of shape (seq_length, VAE.z_dim). In the baseline
                    model, this will contain one tuple from the VAE and another
                    from the Transformer.

    Returns:
    - BCEs: a list containing the total BCE reconstruction loss for each image
            sequence
    - KLDs: a list containing the total KL divergence loss for each mu/sigma pair
    """
    
    # List to aggregate binary cross-entropy reconstruction losses
    # from all of the image outputs:
    BCEs = []
    # List to aggregate KL divergence losses from each of the mu/sigma
    # projections:
    KLDs = []
    
    
    

    # TODO Your code goes here.
    num_methods = len(gen_images)
    for i in range(num_methods):
        
        gen_seqs = gen_images[i]
        
        for gen_seq, target_seq in zip(gen_seqs, input_images):
            gen_seq = gen_seq.view(-1)
            target_seq = target_seq.view(-1)

            bce_loss = F.binary_cross_entropy(gen_seq, target_seq, reduction='mean')
            
            BCEs.append(bce_loss)

        mu, sigma = mu_sigmas[i]
        mu = mu.view(-1)
        sigma = sigma.view(-1)
        log_var = torch.log(sigma * sigma)
        #print("mu:", mu)
        #print("sigma:", sigma)
        #print("log_var", log_var)

        kld_loss = kld(mu, log_var)
        #print("kld_loss", kld_loss)
        
        KLDs.append(kld_loss)
    

    return BCEs, KLDs
