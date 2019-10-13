from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from math import pi

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary', 'deep-GMM'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu


        # GMM Parameters
        self.eps = 1.e-6
        self.n_components = 10
        self.n_features = 32

        self.mu_test  = torch.tensor(np.float32(np.zeros([1, self.n_components, self.n_features])), device=self.device)
        self.var_test = torch.tensor(np.float32(np.ones([1, self.n_components, self.n_features])), device=self.device)



        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = self.to_var(self.phi)
        if mu is None:
            mu = self.to_var(self.mu)
        if cov is None:
            cov = self.to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + self.to_var(torch.eye(D)*eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov).cuda()
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)


        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs, category, resconstruction = net(inputs)
                if self.objective == 'deep-GMM':

                    phi, mu, cov = self.compute_gmm_params(outputs, category)
                    sample_energy, cov_diag = self.compute_energy(outputs, phi=phi, mu=mu, cov=cov,
                                                                size_average=True)

                                                                        # (n,k) --> (n,k,1)
                    # weights = category.unsqueeze(2)
                    #
                    # # (n, k, 1) --> (1, k, 1)
                    # n_k = torch.sum(weights, 0, keepdim=True)
                    #
                    #
                    # # (n,d) ---> (n, k, d)
                    # outputs = outputs.unsqueeze(1).expand(outputs.size(0), self.n_components, outputs.size(1))
                    #
                    # # (n, k, d) --> (1, k, d)
                    # mu = torch.div(torch.sum(weights * outputs, 0, keepdim=True), n_k + self.eps)
                    # var = torch.div(torch.sum(weights * (outputs - mu) * (outputs - mu), 0, keepdim=True), n_k + self.eps)
                    #
                    # self.mu_test = 0.95*self.mu_test + 0.05*mu
                    # self.var_test = 0.95*self.var_test + 0.05*var
                    #
                    # # (1, k, d) --> (n, k, d)
                    # mu = mu.expand(outputs.size(0), self.n_components, self.n_features)
                    # var = var.expand(outputs.size(0), self.n_components, self.n_features)
                    #
                    #
                    #
                    #
                    # # (n, k, d) --> (n, k, 1)
                    # exponent = torch.exp(-.5 * torch.sum((outputs - mu) * (outputs - mu) / var, 2, keepdim=True))
                    # # (n, k, d) --> (n, k, 1)
                    # prefactor = torch.rsqrt(((2. * pi) ** self.n_features) * torch.prod(var, dim=2, keepdim=True) + self.eps)
                    #
                    # # (n, k, 1)
                    # logits_pre = torch.mean(weights, 0, keepdim=True)*prefactor * exponent
                    #
                    # # (n, k, 1) --> (n, k)
                    #
                    # logits_pre = torch.squeeze(logits_pre)


                    #logits = -torch.mean(torch.log(torch.sum(logits_pre, 1) + self.eps))

                    rescon_error = torch.sum((resconstruction - inputs) ** 2, dim=tuple(range(1, resconstruction.dim())))

                    rescon_loss = torch.mean(rescon_error)

                    loss = Variable(sample_energy+rescon_loss, requires_grad= True)


                elif self.objective == 'soft-boundary':
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs,category,resconstruction = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                if self.objective == 'deep-GMM':
                    # (n,k) --> (n,k,1)
                    weights = category.unsqueeze(2)

                    # (n, k, 1) --> (1, k, 1)
                    n_k = torch.sum(weights, 0, keepdim=True)

                    # (n,d) ---> (n, k, d)
                    outputs = outputs.unsqueeze(1).expand(outputs.size(0), self.n_components, outputs.size(1))

                    # (n, k, d) --> (1, k, d)
                    mu = torch.div(torch.sum(weights * outputs, 0, keepdim=True), n_k + self.eps)
                    # (n, k, d) --> (1, k, d)
                    var = torch.div(torch.sum(weights * (outputs - mu) * (outputs - mu), 0, keepdim=True),
                                    n_k + self.eps)

                    # (1, k, d) --> (n, k, d)
                    mu = mu.expand(outputs.size(0), self.n_components, self.n_features)
                    var = var.expand(outputs.size(0), self.n_components, self.n_features)

                    #------------------save mu-?-----------------------

                    # mu = self.mu_test
                    # var = self.var_test
                    #------------------------------------------




                    # (n, k, d) --> (n, k, 1)
                    exponent = torch.exp(-.5 * torch.sum((outputs - mu) * (outputs - mu) / var, 2, keepdim=True))
                    # (n, k, d) --> (n, k, 1)
                    prefactor = torch.rsqrt(
                        ((2. * pi) ** self.n_features) * torch.prod(var, dim=2, keepdim=True) + self.eps)

                    # (n, k, 1)
                    logits_pre = torch.mean(weights, 0, keepdim=True) * prefactor * exponent

                    # (n, k, 1) --> (n, k)
                    logits_pre = torch.squeeze(logits_pre)

                    logits = -torch.log(torch.sum(logits_pre, 1) + self.eps)

                    rescon_error = torch.sum((resconstruction - inputs) ** 2, dim=tuple(range(1, resconstruction.dim())))

                    #scores = logits + rescon_error
                    scores = logits
                    #scores = rescon_error
                elif self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs,_,_ = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
