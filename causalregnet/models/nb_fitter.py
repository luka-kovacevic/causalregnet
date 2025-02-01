import lightning.pytorch as pl
import torch


class NegativeBinomialFitter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.r = torch.nn.Parameter(torch.tensor(1.0))
        self.p = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, observed_expression):
        r = torch.relu(self.r) + 1e-5
        p = torch.sigmoid(self.p)

        nll = -torch.sum(
            (
                torch.lgamma(observed_expression + r)
                - torch.lgamma(r)
                - torch.lgamma(observed_expression + 1)
            )
            + r * torch.log(1 - p)
            + observed_expression * torch.log(p)
        )

        # nll = torch.lgamma(observed_expression + theta) -
        # torch.lgamma(observed_expression + 1) -
        # gammaln(theta) + theta * np.log(theta / (theta + mu)) +
        # observed_expression * np.log(mu / (theta + mu))

        return nll

    def training_step(self, batch, batch_idx):
        data = batch[0]
        loss = self(data)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Using Adam optimizer
        return torch.optim.Adam([self.r, self.p], lr=0.01)
