r"""2d gaussian mixture for images."""
from collections.abc import Generator

import torch
from torch import Size, Tensor, nn
from torch.optim import Adam


class GaussianSplats(nn.Module):
    r"""Gaussian Splatting modules for Image rendering."""

    def __init__(self, n_gaussians: int, n_channels: int = 3) -> None:
        r"""Initialize gaussians."""
        super().__init__()
        self.n_gaussians= n_gaussians
        self.n_channels = None # Assuming RGB colors
        self.alpha_channel = False #future option

        # Initialize the parameters of the Gaussian components
        self.centers = nn.Parameter(2*torch.rand(n_gaussians, 2)-1)
        self.covs = nn.Parameter(torch.randn(n_gaussians, 3).abs()) # (a, b, \theta)
        self._colors = nn.Parameter(torch.randn(n_gaussians, n_channels))
        self._alphas = nn.Parameter(torch.randn(n_gaussians)) # The alpha values for each ellipse
        self.img_size = None

    @property
    def colors(self)-> nn.Parameter:
        r"""Return colors defined between 0 and 1."""
        return self._colors.tanh()

    @property
    def alphas(self)->nn.Parameter:
        r"""Return transparency defined between 0 and 1."""
        return self._alphas.tanh()

    def forward(self, img_size: Size | None = None)-> Tensor:
        r"""Reconstruct the image based on the Gaussians parameters."""
        img_size = self.img_size if img_size is None else img_size
        assert img_size is not None, "Provide an image size or fit before rendering"
        x ,y = torch.linspace(-1,1, img_size[0]), torch.linspace(-1,1, img_size[1])
        x, y = x.to(self.centers.device), y.to(self.centers.device)
        xx, yy = torch.meshgrid(x, y)
        xx = xx.unsqueeze(0).repeat(self.n_gaussians,1,1)
        yy = yy.unsqueeze(0).repeat(self.n_gaussians,1,1)
        # Centering
        xx = xx - self.centers[:,0, None, None]
        yy = yy - self.centers[:,1, None, None]
        # Apply rotation
        theta = self.covs[:,0, None, None]
        rot_xx = xx*theta.cos() - yy*theta.sin()
        rot_yy = xx*theta.sin() + yy*theta.cos()
        # Apply anisotropic scaling
        inv_a = 1/(self.covs[:,2].abs()+1e-7)[:,None,None]
        inv_b = 1/(self.covs[:,1].abs()+1e-7)[:,None,None]
        zz = (rot_xx*inv_a)**2 + (rot_yy*inv_b)**2
        gs = torch.exp(-0.5 * zz)#/(2*torch.pi) *inv_a**2*inv_b**2*
        # gs = torch.relu(0.5*zz) # trick to avoid propagation of nan
        # gs = (zz<=1).float() # ellipses only for stabiltiy
        gs = gs.unsqueeze(1) # prepares for channels wise multiplication
        alphas = self.alphas[:,None,None,None]
        colors = self.colors[:,:,None,None]
        return (alphas*colors*gs).sum(0)/(alphas*gs).sum(0)


    def _fit_iterable(self, img:Tensor, n_iters:int = 100)->Generator[Tensor]:
        r"""Fit the Gaussian parameters to the input image and return intermediate.

        Args:
            img (Tensor): The input image tensor of shape (C, H, W).
        """
        self.img_size = img.shape[1:]
        assert self.colors.shape[-1] == img.shape[0], "Colors mismatch."
        optim=Adam([self.centers, self.covs, self._colors, self._alphas])
        for _ in range(n_iters):
            rendered = self.forward()
            err = torch.nn.MSELoss()(rendered,img)
            print(f"Iteration {_}, Fitting error: {err.item()}", end="\r")
            yield rendered, err
            err.backward()
            optim.step()
            optim.zero_grad()

    def fit(self,img:Tensor, n_iters:int = 100)->None:
        r"""Fit the Gaussian parameters to the image."""
        for i, _ in enumerate(self._fit_iterable(img, n_iters)):
            print(f"Iteration {i}", end="\r")
        print("Done.")



