import matplotlib.pyplot as plt
import matplotlib.pylab as pyl 
from matplotlib.colors import LogNorm
import os
import torch
import numpy as np
from scipy.ndimage import uniform_filter

def plot_submission(samples, reward_fn, div: str, output_dir: str, step: int, range_lim: list, rho_expert=None):
    n_pts = 64 # 0.001   

    # construct test points
    test_grid = setup_grid(range_lim, n_pts)

    # plot
    ims = []
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
    axs = axs.reshape(-1)
    ims.append(plot_reward_fn(axs[0], test_grid, n_pts, reward_fn))
    ims.append(plot_samples(samples, axs[1], range_lim, n_pts))

    # fig.colorbar(im, ax=ax)

    # format
    # for ax, im in zip([axs[0], axs[1], axs[2], axs[3]], ims):
    #     fig.colorbar(im, ax=ax)
    for idx, ax in enumerate(axs):
        format_ax(ax, range_lim)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'plt/vis_step_{step:06}.png')) 
    plt.close()

def plot_reward_traj(reward_fn, output_dir, step, range_lim):
    n_pts = 64 # 0.001   

    # construct test points
    test_grid = setup_grid(range_lim, n_pts)

    # plot
    ims = []
    fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    # axs = axs.reshape(-1)
    im = plot_reward_fn(axs, test_grid, n_pts, reward_fn)

    fig.colorbar(im, ax=axs)
    plt.savefig(os.path.join(output_dir, f'plt/rew_step_{step:06}.png')) 
    plt.close()

def plot_samples_traj(samples, output_dir, step, range_lim):
    n_pts = 64 # 0.001   

    # construct test points
    test_grid = setup_grid(range_lim, n_pts)

    # plot
    ims = []
    fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    # axs = axs.reshape(-1)
    im = plot_samples(samples, axs, range_lim, n_pts)

    fig.colorbar(im, ax=axs)

    plt.savefig(os.path.join(output_dir, f'plt/st_step_{step:06}.png')) 
    plt.close()



def setup_grid(range_lim, n_pts):
    x = torch.linspace(range_lim[0][0], range_lim[0][1], n_pts)
    y = torch.linspace(range_lim[1][0], range_lim[1][1], n_pts)
    xx, yy = torch.meshgrid((x, y))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz

def format_ax(ax, range_lim):
    ax.set_xlim(range_lim[0][0], range_lim[0][1])
    ax.set_ylim(range_lim[1][0], range_lim[1][1])

def plot_samples(samples, ax, range_lim, n_pts):
    # s = samples.reshape(-1, samples.shape[2])
    s = np.array(samples)
    
    indices = np.random.choice(s.shape[0], size=min(10000, s.shape[0]), replace=False)
    s = np.array(s[indices, :])
    
    im = ax.hist2d(x=s[:,0], y=s[:,1], density=True, norm=LogNorm(),
                    range=range_lim, 
                    bins=n_pts, cmap=plt.cm.jet)
    # ax.set_title('S Density')
    ax.set_aspect('equal', 'box')
    return im[3] # https://stackoverflow.com/a/42388179/9072850

def plot_traj(samples, ax, reward_func=None):
    indices = np.random.choice(samples.shape[0], size=min(15, samples.shape[0]), replace=False)
    s = samples[indices]
    rewards = reward_func(s)
    rewards = rewards.reshape(s.shape[0], s.shape[1])
    rewards = rewards.sum(1)
    colors = pyl.cm.turbo(np.linspace(0, 1, 10))

    diff_array = np.absolute(rewards[:, None] - np.linspace(0, 1, 10)*200)
    indexes = np.argmin(diff_array, 1)
    # indexes = np.argsort(rewards)
    # print(indexes)
    # print(colors[indexes])
    # for i, traj in enumerate(s):
    for i, ind in enumerate(indexes):
        traj = s[i]
        ax.plot(traj[:, 0], traj[:, 1], color=colors[ind])
    ax.set_title('Trajectories')
    ax.set_aspect('equal', 'box')

def plot_expert(ax, test_grid, n_pts, rho_expert, title='Expert Density'):
    xx, yy, zz = test_grid
    rho = rho_expert(zz)

    im = ax.pcolormesh(xx, yy, rho.reshape(n_pts,n_pts), norm=LogNorm(), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return im

def plot_agent(ax, test_grid, n_pts, agent_density, title='Agent Density'):
    xx, yy, zz = test_grid
    # print(agent_density)
    rho = np.exp(agent_density(zz))


    im = ax.pcolormesh(xx, yy, rho.reshape(n_pts,n_pts), norm=LogNorm(), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return im

def plot_reward_fn(ax, test_grid, n_pts, reward_fn, title='Reward Map'):
    xx, yy, zz = test_grid
    rewards = reward_fn(zz.numpy())

    im = ax.pcolormesh(xx, yy, rewards.reshape(n_pts,n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    # ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return im