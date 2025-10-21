from glob import glob
import os
import torch
from tqdm import tqdm
from utils import get_fid, interpolate_latent_space, save_plot
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import VisionDataset


def build_transforms():
    # 1. Convert input image to tensor.
    # 2. Rescale input image from [0., 1.] to be between [-1., 1.].
    rescaling = lambda x: (x - 0.5) * 2.0
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # Get optimizers and learning rate schedulers.
    optim_discriminator = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0, 0.9))
    optim_generator = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0, 0.9))
    ##################################################################
    # TODO 1.2: Construct the learning rate schedulers for the
    # generator and discriminator. The learning rate for the
    # discriminator should be decayed to 0 over 500K iterations.
    # The learning rate for the generator should be decayed to 0 over
    # 100K iterations.
    ##################################################################
    scheduler_discriminator = torch.optim.lr_scheduler.CosineAnnealingLR(optim_discriminator, T_max=5e5)
    scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingLR(optim_generator, T_max=1e5)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
    amp_enabled=True,
):
    torch.backends.cudnn.benchmark = True # speed up training
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="../datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)

    scaler = torch.cuda.amp.GradScaler()

    iters = 0
    fids_list = []
    iters_list = []
    pbar = tqdm(total = num_iterations)
    while iters < num_iterations:
        for train_batch in train_loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                train_batch = train_batch.cuda()
                
                ####################### UPDATE DISCRIMINATOR #####################
                ##################################################################
                # TODO 1.2: compute generator, discriminator, and interpolated outputs
                # 1. Compute generator output
                # Note: The number of samples must match the batch size.
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.
                ##################################################################
                num_samples = train_batch.shape[0] * 128
                gen_output = gen(num_samples) # (batch_size, 128, 32, 32)
                discrim_real = disc(train_batch) # (batch_size, 1)
                discrim_fake = disc(gen_output) # (batch_size, 1)
                ##################################################################
                #                          END OF YOUR CODE                      #
                ##################################################################

                ##################################################################
                # TODO 1.5 Compute the interpolated batch and run the
                # discriminator on it.
                ###################################################################
                epson = torch.rand(train_batch.shape[0],1, 1, 1)  # (batch_size, 1, 1, 1)
                epson = epson.cuda()
                # print("epson device is {}, train batch device is {}, gen_output device is ".format(epson.device, train_batch.device, gen_output.device))
                interp = epson * train_batch + (1 - epson) * gen_output  # (batch_size, 128, 32, 32)
                discrim_interp = disc(interp) # (batch_size, 1)
                ##################################################################
                #                          END OF YOUR CODE                      #
                ##################################################################

            discriminator_loss = disc_loss_fn(
                discrim_real, discrim_fake, discrim_interp, interp, lamb
            )
            
            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward()
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()

            if iters % 5 == 0:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    ##################################################################
                    # TODO 1.2: Compute generator and discriminator output on
                    # generated data.
                    ###################################################################
                    num_samples = batch_size * 128
                    fake_batch = gen(num_samples) # (batch_size, 128, 32, 32)
                    discrim_fake = disc(fake_batch) # (batch_size, 1)
                    ##################################################################
                    #                          END OF YOUR CODE                      #
                    ##################################################################

                    generator_loss = gen_loss_fn(discrim_fake)

                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()

            if iters % log_period == 0 and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        ##################################################################
                        # TODO 1.2: Generate samples using the generator.
                        # Make sure they lie in the range [0, 1]!
                        ##################################################################
                        num_samples = batch_size * 128
                        generated_samples = gen(num_samples) # (batch_size, 128, 32, 32)
                        generated_samples = generated_samples.clamp(min=0,max=1)
                        ##################################################################
                        #                          END OF YOUR CODE                      #
                        ##################################################################
                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    if os.environ.get('PYTORCH_JIT', 1):
                        torch.jit.save(torch.jit.script(gen), prefix + "/generator.pt")
                        torch.jit.save(torch.jit.script(disc), prefix + "/discriminator.pt")
                    else:
                        torch.save(gen, prefix + "/generator.pt")
                        torch.save(disc, prefix + "/discriminator.pt")
                    fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=256,
                        num_gen=10_000,
                    )
                    print(f"Iteration {iters} FID: {fid}")
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
            pbar.update(1)
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=256,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")
