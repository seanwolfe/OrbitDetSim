import torch
import torch.nn as nn
import torch.autograd as autograd
import spiceypy as spice
import numpy as np


# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')

class ELM(nn.Module):
    def __init__(self, hidden_dim, q=3, activation=torch.tanh, c_normalization=1.0):
        """
        :param input_dim: Dimensionality of each input vector (e.g., 1 for time).
        :param hidden_dim: Number of hidden neurons (N_star).
        :param q: Number of output dimensions (e.g., 3 for 3D position).
        :param activation: Activation function, e.g., tanh.
        """
        super().__init__()
        input_dim= 1
        self.input_weights = nn.Parameter(torch.randn(hidden_dim, input_dim), requires_grad=False)  # (H x 1)
        self.bias = nn.Parameter(torch.randn(hidden_dim), requires_grad=False)  # (H,)
        self.activation = activation
        self.output_weights = nn.Parameter(torch.randn(q, hidden_dim))  # (q x H)
        self.c_normalization = c_normalization

    def forward(self, z):
        # z should be of shape (d, input_dim), typically (d, 1)
        # Transpose z to (input_dim, d) so matmul (H x 1) @ (1 x d) => (H x d)
        H = self.activation(self.input_weights @ z.T + self.bias[:, None])  # (H x d)
        Y = self.output_weights @ H  # (q x H) @ (H x d) => (q x d)
        return Y.T  # Return shape (d x q)

    def forward_with_derivatives(self, z):
        """
        z: shape (d, 1)
        Returns: y, y', y'' each of shape (d, q)
        """
        W = self.input_weights  # (H x 1)
        b = self.bias[:, None]  # (H x 1)
        z = W @ z.T + b  # (H x d)
        H = torch.tanh(z)  # (H x d)

        H_prime = (1 - H ** 2) * W  # (H x d)
        H_double_prime = -2 * H * (1 - H ** 2) * (W ** 2)  # (H x d)

        Y = self.output_weights @ H  # (q x N)
        Y_dot = self.c_normalization * self.output_weights @ H_prime  # (q x d)
        Y_dot_dot = self.c_normalization ** 2 * self.output_weights @ H_double_prime  # (q x d)

        return Y.T, Y_dot.T, Y_dot_dot.T


def nbody_physics_loss(Y_pred, Y_ddot_pred, epochs, configuration):
    """
    Y_pred: (N, 3) — predicted positions
    Y_ddot_pred: (N, 3) — predicted accelerations
    positions: (N, n, 3) — positions of other bodies at each epoch
    masses: (n,) — masses of influencing bodies
    """

    def get_nbody_positions(epochs, config):
        """
        :param epochs: numpy array of shape (N,) — JDTDB times
        :param config: dictionary containing masses
        :return: positions (N, n, 3), masses (n,)
        """
        bodies = [10, 1, 2, 399, 4, 5, 6, 7, 8, 301]  # SPICE IDs: SUN, MERCURY, ..., MOON

        masses = np.array([config[f'{name}_MASS'] for name in
                           ['SUN', 'MERCURY', 'VENUS', 'EARTH', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE',
                            'MOON']])

        N = len(epochs)
        n = len(bodies)
        positions = np.zeros((N, n, 3))  # Output array

        # Convert epochs from JDTDB to ET
        epoch_ets = spice.unitim(epochs, 'JDTDB', 'ET')  # (N,)

        for i, et in enumerate(epoch_ets):
            for j, body in enumerate(bodies):
                state, _ = spice.spkgeo(targ=body, et=et, ref='ECLIPJ2000', obs=10)  # observer is Sun (10)
                positions[i, j, :] = state[:3]  # only position

        return torch.tensor(positions, dtype=torch.float32), torch.tensor(masses, dtype=torch.float32)

    positions, masses = get_nbody_positions(epochs, configuration)
    G = configuration['GRAVITATIONAL_CONSTANT']

    N, n, _ = positions.shape
    Y_expanded = Y_pred[:, None, :]  # (N, 1, 3)

    r_vecs = positions - Y_expanded  # (N, n, 3)
    r_norms = torch.norm(r_vecs, dim=-1, keepdim=True)  # (N, n, 1)
    accel_terms = G * masses[None, :, None] * r_vecs / (r_norms ** 3 + 1e-9)  # (N, n, 3)  # in km

    total_accel = accel_terms.sum(dim=1)  # (N, 3)
    return torch.mean((Y_ddot_pred - total_accel) ** 2)


# Training
def train(model, x_data, y_data, epochs=1000, lr=1e-2, lambda_phys=1.0):
    optimizer = torch.optim.Adam([model.output_weights], lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_data)
        data_loss = torch.mean((pred - y_data) ** 2)
        phys_loss = physics_loss(model, x_data)
        loss = data_loss + lambda_phys * phys_loss
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Data Loss = {data_loss.item():.4e}, Physics Loss = {phys_loss.item():.4e}")


# elm = ELM(hidden_dim=50, q=3)
# z = torch.linspace(0, 1, 100).unsqueeze(1)  # (100 x 1)
# y_pred = elm(z)  # (100 x 3)
