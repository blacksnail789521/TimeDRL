import numpy as np
import torch


def jitter(x, sigma=0.1):
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)


def scaling(x, sigma=1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(
                    x.shape[1] - 2, num_segs[i] - 1, replace=False
                )
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def masking(x, percentage=0.3):
    mask = np.random.choice(
        [0, 1],
        size=(x.shape[0], x.shape[2], x.shape[1]),
        p=[percentage, 1 - percentage],
    )
    return np.transpose(np.multiply(np.transpose(x, (0, 2, 1)), mask), (0, 2, 1))


def cropping(x, min_percentage=0.5, max_percentage=1.0):
    cropped = np.zeros_like(x)
    for i, pat in enumerate(x):
        length = pat.shape[0]
        crop_length = np.random.randint(
            int(length * min_percentage), int(length * max_percentage)
        )
        start = np.random.randint(0, length - crop_length)
        end = start + crop_length
        cropped[i, start:end, :] = pat[start:end, :]
    return cropped


def data_augmentation(x, method, **kwargs):
    # Change from tensor to numpy
    device = x.device
    dtype = x.dtype
    x = x.cpu().detach().numpy()

    # Apply augmentation
    if method == "jitter":
        x_aug = jitter(x, **kwargs)
    elif method == "scaling":
        x_aug = scaling(x, **kwargs)
    elif method == "rotation":
        x_aug = rotation(x, **kwargs)
    elif method == "permutation":
        x_aug = permutation(x, **kwargs)
    elif method == "masking":
        x_aug = masking(x, **kwargs)
    elif method == "cropping":
        x_aug = cropping(x, **kwargs)
    else:
        raise NotImplementedError
    assert x_aug.shape == x.shape

    # Change from numpy to tensor
    x_aug = torch.from_numpy(x_aug).to(device=device, dtype=dtype)

    return x_aug
