import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GMM(nn.Module):
    def __init__(self, N_GM, LATENT_DIM):
        super(GMM, self).__init__()
        means = torch.linspace(-0.5, 0.5, N_GM).unsqueeze(1)
        std = torch.linspace(-0.5, 0.5, N_GM).unsqueeze(1)
        self.com_mu = torch.nn.Parameter(
            torch.randn(N_GM, LATENT_DIM, requires_grad=True) + means)
        self.com_sigma = torch.nn.Parameter(
            torch.randn(N_GM, LATENT_DIM, requires_grad=True) + std)

    def forward(self):
        return self.com_mu, self.com_sigma


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class StdMap1d(nn.Module):
    def __init__(self):
        super(StdMap1d, self).__init__()

    def forward(self, x):
        std = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt((std ** 2).mean(dim=0) + 1e-6).mean()
        std_map = std.expand(x.size(0), 1, x.size(2))
        x = torch.cat((x, std_map), dim=1)
        return x


def hyper_generator(com_mu_sig, hyper_k, hyper_noise):
    com_mu, com_sig = com_mu_sig()
    noise_mean = torch.matmul(hyper_k, com_mu)
    noise_log_var = torch.matmul(hyper_k, com_sig)
    noise_std = torch.exp(noise_log_var*0.5)

    noise = torch.add(torch.mul(noise_std, hyper_noise), noise_mean)

    return noise


class NetUp(nn.Module):
    def __init__(self, d, echannels=23, out=False, spectralnorm=False):
        super(NetUp, self).__init__()

        self.out = out
        self.spectralnorm = spectralnorm

        self.net_up_middle_block_spectralnorm = nn.Sequential(
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=1,
                                    padding=3, padding_mode='reflect', bias=True)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=1,
                                    padding=2, padding_mode='reflect', bias=True)),
            nn.LeakyReLU(0.1),
        )

        self.net_up_middle_block_batchnorm = nn.Sequential(
            nn.Upsample(scale_factor=2),
            (nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=1,
                       padding=3, padding_mode='reflect', bias=False)),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.1),
            (nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=1,
                       padding=2, padding_mode='reflect', bias=False)),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.1),
        )

        self.net_up_output = nn.Sequential(
            (nn.Conv1d(d * 4, echannels,
                       kernel_size=1, stride=1, bias=True)),
            nn.Tanh()
        )

    def forward(self, x):
        if self.spectralnorm:
            middle = self.net_up_middle_block_spectralnorm(x)
        else:
            middle = self.net_up_middle_block_batchnorm(x)

        if self.out:
            out = self.net_up_output(middle)
            return middle, out
        else:
            return middle


class NetDown(nn.Module):
    def __init__(self, d, std=False, spectralnorm=False):
        super(NetDown, self).__init__()

        self.std = std

        self.spectralnorm = spectralnorm

        self.minibatchstd = StdMap1d()

        self.net_down_middle_block_spectralnorm = nn.Sequential(
            spectral_norm(
                nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=1, padding=3, padding_mode='reflect', bias=True)),
            nn.LeakyReLU(0.1),
            spectral_norm(
                nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=2, padding=2, padding_mode='reflect', bias=True)),
            nn.LeakyReLU(0.1),
            # nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.net_down_middle_block_spectralnorm_std = nn.Sequential(
            spectral_norm(
                nn.Conv1d(d * 4+1, d * 4, kernel_size=6, stride=1, padding=3, padding_mode='reflect', bias=True)),
            nn.LeakyReLU(0.1),
            spectral_norm(
                nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=2, padding=2, padding_mode='reflect', bias=True)),
            nn.LeakyReLU(0.1),
            # nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.net_down_middle_block_batchnorm = nn.Sequential(
            (
                nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=1, padding=3, padding_mode='reflect', bias=False)),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.1),
            (
                nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=2, padding=2, padding_mode='reflect', bias=False)),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.1),
            # nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        if self.std:
            x = self.minibatchstd(x)
            middle = self.net_down_middle_block_spectralnorm_std(x)

        else:
            if self.spectralnorm:
                middle = self.net_down_middle_block_spectralnorm(x)
            else:
                middle = self.net_down_middle_block_batchnorm(x)

        return middle


class GeneratorG(nn.Module):
    def __init__(self, h_size=128, v_size=32, d=16, c_size=6, echannels=23):
        super().__init__()
        self.c_size = c_size

        self.netG_up0 = nn.Sequential(
            nn.Linear(h_size + v_size + c_size, d * 4 * 16),
            nn.LeakyReLU(0.1),
            Reshape(-1, d * 4, 16),
        )

        self.netG_up1 = NetUp(d)
        self.netG_up2 = NetUp(d)
        self.netG_up3 = NetUp(d)
        self.netG_up4 = NetUp(d, out=True)
        # self.netG_up5 = NetUp(d, out=True)

    def forward(self, h, vt, c):
        if self.c_size > 0:
            xt = torch.cat([h, vt, c], dim=1)
        else:
            xt = torch.cat([h, vt], dim=1)

        xt = self.netG_up0(xt)

        xt = self.netG_up1(xt)
        xt = self.netG_up2(xt)
        xt = self.netG_up3(xt)
        xt, out = self.netG_up4(xt)
        # xt, out = self.netG_up5(xt)

        return out


# md = GeneratorG()
# summary(md, input_size=[(128,), (32,)])


class Discriminator1(nn.Module):
    def __init__(self, h_size=128, v_size=32, c_size=6, d=16, echannels=23):
        super().__init__()
        self.c_size = c_size

        self.netD1_down0 = nn.Sequential(
            (
                spectral_norm(nn.Conv1d(echannels, d * 4, kernel_size=1, stride=1, bias=True))),
            nn.LeakyReLU(0.1),
        )

        self.netD1_down1 = NetDown(d, spectralnorm=True)
        self.netD1_down2 = NetDown(d, spectralnorm=True)
        self.netD1_down3 = NetDown(d, spectralnorm=True)
        self.netD1_down4 = NetDown(d, spectralnorm=True, std=True)
        # self.netD1_down5 = NetDown(d, spectralnorm=True)

        self.netD1_fc0 = nn.Sequential(
            spectral_norm(nn.Linear(h_size + v_size + c_size, 256)), nn.LeakyReLU(0.1))
        self.netD1_fc1 = nn.Sequential(
            spectral_norm(nn.Linear(1536 + 256, 512)), nn.LeakyReLU(0.1))
        self.netD1_fc2 = spectral_norm(nn.Linear(512, 1))

    def forward(self, xt, h, vt, c):
        xt = self.netD1_down0(xt)

        xt = self.netD1_down1(xt)
        xt = self.netD1_down2(xt)
        xt = self.netD1_down3(xt)
        xt = self.netD1_down4(xt)
        # xt = self.netD1_down5(xt)
        xt = torch.flatten(xt, start_dim=1)

        if self.c_size > 0:
            hvt = torch.cat([h.squeeze(), vt.squeeze(), c.squeeze()], dim=1)
        else:
            hvt = torch.cat([h.squeeze(), vt.squeeze()], dim=1)
        hvt = self.netD1_fc0(hvt)

        xthvt = torch.cat([xt, hvt], dim=1)
        xthvt = self.netD1_fc1(xthvt)

        # return self.netD1_fc2(xthvt), self.netD1_fc_(xt)
        return self.netD1_fc2(xthvt)

# md = Discriminator1()
# summary(md, input_size=[(23, 512), (128,1), (32,1)])


class Extractor1(nn.Module):
    def __init__(self, h_size=128, d=16, c_size=3, echannels=23, T=5):
        super().__init__()
        self.c_size = c_size

        self.netE1_down0 = nn.Sequential(
            (
                nn.Conv1d(echannels * T, d * 4, kernel_size=1, stride=1, bias=True)),
            nn.LeakyReLU(0.1),
        )

        self.netE1_down1 = NetDown(d)
        self.netE1_down2 = NetDown(d)
        self.netE1_down3 = NetDown(d)
        self.netE1_down4 = NetDown(d)
        # self.netE1_down5 = NetDown(d)

        self.fcE1_mu = (nn.Linear(1536+c_size, h_size))
        self.fcE1_sig = (nn.Linear(1536+c_size, h_size))

    def forward(self, x_T, device, c):
        h = torch.cat(x_T, 1)  # concat along channels
        h = self.netE1_down0(h)

        h = self.netE1_down1(h)
        h = self.netE1_down2(h)
        h = self.netE1_down3(h)
        h = self.netE1_down4(h)
        # ht = self.netE1_down5(ht)
        h = torch.flatten(h, start_dim=1)
        if self.c_size > 0:
            h = torch.cat([h, c], dim=1)

        mean = self.fcE1_mu(h)

        log_var = self.fcE1_sig(h)
        std = torch.exp(log_var*0.5)

        eps = torch.randn(*std.shape, device=device)

        h = torch.add(mean, torch.mul(std, eps))
        return h, mean, std


# md = Extractor1()
# summary(md, input_size=(23 * 30, 512))


class DynamicExtractor2(nn.Module):
    def __init__(self, v_size=32, d=16, echannels=23):
        super().__init__()

        self.netE2_down0 = nn.Sequential(
            (
                nn.Conv1d(echannels, d * 4, kernel_size=1, stride=1, bias=True)),
            nn.LeakyReLU(0.1),
        )

        self.netE2_down1 = NetDown(d)
        self.netE2_down2 = NetDown(d)
        self.netE2_down3 = NetDown(d)
        self.netE2_down4 = NetDown(d)
        # self.netE2_down5 = NetDown(d)

        self.fcE2 = (nn.Linear(1536, v_size))

    def forward(self, xt):
        vt = self.netE2_down0(xt)

        vt = self.netE2_down1(vt)
        vt = self.netE2_down2(vt)
        vt = self.netE2_down3(vt)
        vt = self.netE2_down4(vt)
        # vt = self.netE2_down5(vt)
        vt = torch.flatten(vt, start_dim=1)

        return self.fcE2(vt)

# md = DynamicExtractor2()
# summary(md, input_size=(23, 512))


class DynamicGeneratorO(nn.Module):
    def __init__(self, v_size=32, e_size=16):
        super().__init__()
        self.netGO = nn.Sequential(
            nn.Linear(v_size + e_size, 512),
            nn.LeakyReLU(0.1),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, v_size)
        )

        # self.res = (nn.Linear(v_size, v_size))

    def forward(self, vt, et):
        vtnext = torch.cat([vt, et], dim=1)
        vtnext = self.netGO(vtnext)
        return vtnext
        # vt = self.res(vt)
        # return vtnext + vt

# md = DynamicGeneratorO()
# summary(md, input_size=[(32,), (16,)])


class Discriminator2(nn.Module):
    def __init__(self, v_size=32):
        super().__init__()
        self.netD2 = nn.Sequential(
            spectral_norm(nn.Linear(v_size * 2, 512)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Linear(512, 512)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Linear(256, 1))
        )

    def forward(self, vt, vtnext):
        vt = torch.cat([vt, vtnext], dim=1)
        return self.netD2(vt)

# md = Discriminator2()
# summary(md, input_size=[(32,), (32,)])


class DiscriminatorGMM(nn.Module):
    def __init__(self, k_size=10, z_size=128):
        super().__init__()
        self.netDGMM = nn.Sequential(
            spectral_norm(nn.Linear(k_size + z_size, 512)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Linear(512, 512)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Linear(256, 1))
        )

    def forward(self, k, z):
        kz = torch.cat([k, z], dim=1)
        return self.netDGMM(kz)


class HyperExtractor(nn.Module):
    def __init__(self, z_size=100, n_gm=10):
        super().__init__()
        self.fcHE = nn.Linear(z_size, n_gm)

    def forward(self, z):
        k = self.fcHE(z)
        k = nn.functional.gumbel_softmax(k, tau=0.1, hard=True)
        return k
