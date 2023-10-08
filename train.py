
import argparse
import time
import datetime
import os
import sys
import numpy as np
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.distributions import OneHotCategorical
from models import *
from data import get_data_loader
from utils import to_device, plot_freq_domain, plot_time_domain, set_seed
import torch.fft as fft

set_seed()

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str,
                    default="gmmarkov-gan", help="name of the experiment")
parser.add_argument("--dataset_prefix", type=str,
                    default="eeg_dataset/", help="path to the train and valid dataset")
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--shuffle", type=bool,
                    default=True, help="shuffle dataset")
parser.add_argument("--is_eval", type=bool,
                    default=False, help="evaluation mode")
parser.add_argument("--gpu_idx", type=int,
                    default=0, help="GPU index")
parser.add_argument("--n_epochs", type=int, default=601,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="adam: learning rate")
parser.add_argument("--lr_disc", type=float, default=4e-4,
                    help="adam: discriminator learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--delta_mome", type=float, default=10,
                    help="Loss weight for moments"),
parser.add_argument("--omega_fft", type=float, default=0.1,
                    help="Loss weight for FFT"),
parser.add_argument("--preload_gpu", type=bool,
                    default=True, help="Preload data to GPU")
parser.add_argument("--sampling_rate", type=int, default=256,
                    help="sampling rate of the signals")
parser.add_argument("--checkpoint_interval", type=int,
                    default=20, help="interval between model checkpoints")

args, unknown = parser.parse_known_args()
print(args)

device = torch.device('cuda:%s' %
                      args.gpu_idx if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
print(cuda, device)

tb = SummaryWriter("logs/%s" % args.experiment_name)

# ------------------

def init_weights(models):
    for model in models:
        model.apply(weights_init_normal)


def viz_histograms(models, epoch):
    for model in models:
        for name, weight in model.named_parameters():
            try:
                tb.add_histogram(name, weight, epoch)
                tb.add_histogram(f'{name}.grad', weight.grad, epoch)
            except NotImplementedError:
                continue


def reset_gradients_to_train(models):
    for model in models:
        model.train()
        for p in model.parameters():
            p.grad = None


def moment_loss(X_p, X_q):
    loss_v1 = criterion_moments(X_p.mean(dim=0), X_q.mean(dim=0).detach())
    loss_v2 = criterion_moments(
        X_p.std(dim=0) + 1e-6, X_q.std(dim=0).detach() + 1e-6)
    loss_v = loss_v1 + loss_v2
    return loss_v


def fft_loss(X_p, X_q):
    fft_fake_x = fft.rfft(X_p)
    fft_abs_fake_x = torch.abs(fft_fake_x).mean(dim=0)
    fft_phase_fake_x = torch.angle(fft_fake_x).mean(dim=0)

    fft_real_x = fft.rfft(X_q)
    fft_abs_real_x = torch.abs(fft_real_x).mean(dim=0)
    fft_phase_real_x = torch.angle(fft_real_x).mean(dim=0)

    loss_fft_abs_x = criterion_fft(fft_abs_fake_x,  fft_abs_real_x)
    loss_fft_phase_x = criterion_fft(fft_phase_fake_x, fft_phase_real_x)

    loss_fft = loss_fft_abs_x + loss_fft_phase_x
    return loss_fft


os.makedirs("saved_models/%s" % args.experiment_name, exist_ok=True)
os.makedirs("sample_signals/%s" % args.experiment_name, exist_ok=True)
os.makedirs("logs/%s" % args.experiment_name, exist_ok=True)

# Load data
dataloader = get_data_loader(args.dataset_prefix, args.batch_size, device=device,
                             shuffle=args.shuffle,
                             preload_gpu=args.preload_gpu)

# Initialize generator and discriminator
N_GM = 10
PI = torch.tensor([1. / N_GM, ] * N_GM)
LATENT_DIM = 128
EPS_DIM = 16
V_DIM = 32
C_SIZE = 6
T = 10
ECHANNELS = 23
d = 24
split_size = args.sampling_rate
val_patient = [0, 1, 0, 0, 0, 0]

prior_k = OneHotCategorical(PI)
com_mu_sig = GMM(N_GM, LATENT_DIM)
generatorG = GeneratorG(h_size=LATENT_DIM, v_size=V_DIM, c_size=C_SIZE, echannels=ECHANNELS, d=d)
generatorO = DynamicGeneratorO(v_size=V_DIM)
extractor1 = Extractor1(h_size=LATENT_DIM, c_size=C_SIZE, T=T, echannels=ECHANNELS, d=d)
extractor2 = DynamicExtractor2(v_size=V_DIM, echannels=ECHANNELS, d=d)
hyper_extractor = HyperExtractor(z_size=LATENT_DIM, n_gm=N_GM)
discriminator1 = Discriminator1(h_size=LATENT_DIM, v_size=V_DIM, c_size=C_SIZE, echannels=ECHANNELS, d=d)
discriminator2 = Discriminator2(v_size=V_DIM)
discriminatorGMM = DiscriminatorGMM(k_size=N_GM, z_size=LATENT_DIM)
models_GE = [com_mu_sig, generatorG, generatorO, extractor1, extractor2, hyper_extractor]
models_D = [discriminator1, discriminator2, discriminatorGMM]
models = models_GE + models_D

criterion = torch.nn.BCEWithLogitsLoss()
criterion_moments = torch.nn.L1Loss()
criterion_fft = torch.nn.L1Loss()

# Optimizers
optimizer_EMB = torch.optim.Adam(
    com_mu_sig.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_G = torch.optim.Adam(
    list(generatorG.parameters()) + list(generatorO.parameters()), lr=args.lr, betas=(args.b1, args.b2))
optimizer_E = torch.optim.Adam(
    list(extractor1.parameters()) + list(extractor2.parameters()), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(
    list(discriminator1.parameters()) + list(discriminator2.parameters()) + list(discriminatorGMM.parameters()), lr=args.lr_disc, betas=(args.b1, args.b2))

if cuda:
    to_device(models, device)
    PI = PI.to(device)
    criterion = criterion.to(device)
    criterion_moments = criterion_moments.to(device)
    criterion_fft = criterion_fft.to(device)
    
if args.epoch != 0:
    # Load pretrained models
    pretrained_path = "saved_models/%s/multi_models_%s.pth" % (args.experiment_name, args.epoch)
    checkpoint = torch.load(pretrained_path, map_location=device)
    com_mu_sig.load_state_dict(checkpoint['com_mu_state_dict'])
    generatorG.load_state_dict(checkpoint['generatorG_state_dict'])
    generatorO.load_state_dict(checkpoint['generatorO_state_dict'])
    extractor1.load_state_dict(checkpoint['extractor1_state_dict'])
    extractor2.load_state_dict(checkpoint['extractor2_state_dict'])
    hyper_extractor.load_state_dict(checkpoint['hyper_extractor_state_dict'])
    discriminator1.load_state_dict(checkpoint['discriminator1_state_dict'])
    discriminator2.load_state_dict(checkpoint['discriminator2_state_dict'])
    discriminatorGMM.load_state_dict(checkpoint['discriminatorGMM_state_dict'])

    optimizer_EMB.load_state_dict(checkpoint['optimizer_EMB_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_E.load_state_dict(checkpoint['optimizer_E_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
else:
    # Initialize weights
    init_weights(models[1:])

prev_time = time.time()
for epoch in range(args.epoch+1, args.n_epochs):

    for i, batch in enumerate(dataloader):
        # Model inputs
        if args.preload_gpu:
            X_q, conds = batch[0], batch[1]
        else:
            X_q = batch.to(device, non_blocking=True).squeeze()

        bs = len(X_q)
        real = torch.full((bs, 1), 1, dtype=torch.float, device=device)
        real_soft = torch.full((bs, 1), 1, dtype=torch.float, device=device)
        fake = torch.full((bs, 1), 0, dtype=torch.float, device=device)

        err_GG_T, err_GO_T, err_E1_T, err_E2_T, err_V_T, err_D1_T, err_D2_T = [], [], [], [], [], [], []

        # ----------------------------
        # Train Discriminators
        # ----------------------------

        reset_gradients_to_train(models_D)

        # GMM
        hyper_noise = torch.randn(bs, LATENT_DIM, device=device)
        k_p = prior_k.sample((bs,)).to(device)

        h_p = hyper_generator(com_mu_sig, k_p, hyper_noise)

        x_T_q = torch.split(X_q, split_size_or_sections=split_size, dim=-1)
        h_q, mu_q, sig_q = extractor1(x_T_q, device, conds)

        k_q = hyper_extractor(h_q)

        fake_validity = discriminatorGMM(k_p.detach(), h_p.detach())
        err_DGMM_fake = criterion(fake_validity, fake)
        err_DGMM_fake.backward()
        real_validity = discriminatorGMM(k_q.detach(), h_q.detach())
        err_DGMM_real = criterion(real_validity, real_soft)
        err_DGMM_real.backward()
        err_DGMM = err_DGMM_real + err_DGMM_fake

        x_T_p = []
        v_T_p = []
        v_T_q = []

        vt_p = torch.randn(bs, V_DIM, device=device)
        xt_q = x_T_q[0]
        vt_q = extractor2(xt_q)
        for idx in range(T):
            xt_p = generatorG(h_p, vt_p, conds)

            x_T_p.append(xt_p)
            v_T_p.append(vt_p)
            v_T_q.append(vt_q)

            # D1
            fake_validity = discriminator1(xt_p.detach(), h_p.detach(), vt_p.detach(), conds)
            err_D1_fake = criterion(fake_validity, fake)
            err_D1_fake.backward()

            real_validity = discriminator1(xt_q.detach(), h_q.detach(), vt_q.detach(), conds)
            err_D1_real = criterion(real_validity, real_soft)
            err_D1_real.backward()
            err_D1_T.append(err_D1_real.item() + err_D1_fake.item())

            if idx < T - 1:
                epst_p = torch.randn(bs, EPS_DIM, device=device)
                vtnext_p = generatorO(vt_p, epst_p)

                xtnext_q = x_T_q[idx + 1]
                vtnext_q = extractor2(xtnext_q)

                # D2
                fake_validity = discriminator2(vt_p.detach(), vtnext_p.detach())
                err_D2_fake = criterion(fake_validity, fake)
                err_D2_fake.backward()

                real_validity = discriminator2(vt_q.detach(), vtnext_q.detach())
                err_D2_real = criterion(real_validity, real_soft)
                err_D2_real.backward()
                err_D2_T.append(err_D2_real.item() + err_D2_fake.item())

                vt_p = vtnext_p
                vt_q = vtnext_q
                xt_q = xtnext_q

        optimizer_D.step()

        # ----------------------------
        # Train Generators and Extractors
        # ----------------------------

        reset_gradients_to_train(models_GE)

        # GMM
        fake_validity_GGMM = discriminatorGMM(k_p, h_p)
        err_GGMM = criterion(fake_validity_GGMM, real)
        err_GGMM.backward(retain_graph=True)

        real_validity_EGMM = discriminatorGMM(k_q, h_q)
        err_EGMM = criterion(real_validity_EGMM, fake)
        err_EGMM.backward(retain_graph=True)

        for idx in range(T):
            # G
            fake_validity_GG = discriminator1(
                x_T_p[idx], h_p, v_T_p[idx], conds)
            err_GG = criterion(fake_validity_GG, real)
            err_GG_T.append(err_GG.item())
            err_GG.backward(retain_graph=True)

            # E
            real_validity_E1 = discriminator1(
                x_T_q[idx], h_q, v_T_q[idx], conds)
            err_E1 = criterion(real_validity_E1, fake)
            err_E1_T.append(err_E1.item())
            err_E1.backward(retain_graph=True)

            if idx < T - 1:
                # G
                fake_validity_GO = discriminator2(v_T_p[idx], v_T_p[idx + 1])
                err_GO = criterion(fake_validity_GO, real)
                err_GO_T.append(err_GO.item())
                err_GO.backward(retain_graph=True)

                # E
                real_validity_E2 = discriminator2(v_T_q[idx], v_T_q[idx + 1])
                err_E2 = criterion(real_validity_E2, fake)
                err_E2_T.append(err_E2.item())
                err_E2.backward(retain_graph=True)

        gen_samples = torch.cat(x_T_p, dim=-1)
        loss_v = moment_loss(gen_samples, X_q) * args.delta_mome
        loss_fft = fft_loss(gen_samples, X_q) * args.omega_fft
        (loss_fft + loss_v).backward()

        optimizer_EMB.step()
        optimizer_G.step()
        optimizer_E.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = args.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [err_GG: %f] [err_GO: %f] [err_E1: %f] [err_E2: %f] [err_D1: %f] [err_D2: %f] [err_DGM: %f] [err_V: %f] [err_F: %f] ETA: %s"
            % (epoch, args.n_epochs, i, len(dataloader), np.mean(err_GG_T), np.mean(err_GO_T), np.mean(err_E1_T), np.mean(err_E2_T),
               np.mean(err_D1_T), np.mean(err_D2_T),  err_DGMM.item(), loss_v, loss_fft.item(), time_left)
        )

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        viz_histograms(models, epoch)

        # Save model checkpoints
        torch.save({
            'epoch': epoch,
            'com_mu_state_dict': com_mu_sig.state_dict(),
            'generatorG_state_dict': generatorG.state_dict(),
            'generatorO_state_dict': generatorO.state_dict(),
            'extractor1_state_dict': extractor1.state_dict(),
            'extractor2_state_dict': extractor2.state_dict(),
            'hyper_extractor_state_dict': hyper_extractor.state_dict(),
            'discriminator1_state_dict': discriminator1.state_dict(),
            'discriminator2_state_dict': discriminator2.state_dict(),
            'discriminatorGMM_state_dict': discriminatorGMM.state_dict(),

            'optimizer_EMB_state_dict': optimizer_EMB.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_E_state_dict': optimizer_E.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),

        }, "saved_models/%s/multi_models_%s.pth" % (args.experiment_name, epoch))

        com_mu_sig.eval()
        generatorG.eval()
        generatorO.eval()
        with torch.no_grad():
            if args.preload_gpu:
                valid_data = next(iter(dataloader))[0]
            else:
                valid_data = next(iter(dataloader))
            valid_data = valid_data.squeeze()[:, 12, :].detach().cpu()

            patient = torch.from_numpy(np.array([val_patient] * args.batch_size)).float().to(device)
            k_p = prior_k.sample((args.batch_size,)).to(device)
            hyper_noise = torch.randn(args.batch_size, LATENT_DIM, device=device)
            h_p = hyper_generator(com_mu_sig, k_p, hyper_noise)
            vt_p = torch.randn(bs, V_DIM, device=device)

            x_T_p = []
            for idx in range(T):
                xt_p = generatorG(h_p, vt_p, patient)
                x_T_p.append(xt_p)
                if idx < T - 1:
                    epst_p = torch.randn(bs, EPS_DIM, device=device)
                    vt_p = generatorO(vt_p, epst_p)

            gen_samples = torch.cat(x_T_p, dim=-1).squeeze()[:, 12, :].detach().cpu()

            img_dir = "sample_signals/%s/time_%s.png" % (args.experiment_name, epoch)
            plot_time_domain(valid_data, gen_samples, img_dir)

            img_dir = "sample_signals/%s/freq_%s.png" % (args.experiment_name, epoch)
            plot_freq_domain(valid_data, gen_samples, args.sampling_rate, img_dir)
