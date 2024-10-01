import numpy as np
import casino
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
import time

patch_size = 14
pca_components = 3
model_name = "dinov2_vits14"

dino_model = torch.hub.load(
    "facebookresearch/dinov2", model=model_name, pretrained=True
).eval()

# device = casino.learning.DEFAULT_TORCH_DEVICE
# casino.hardware.clear_torch_memory()

img_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def forward(image, seg_mask=None):
    frame = {}
    x = img_transform(image)
    _img_size = list(x.shape[1:3])
    _feat_map_size = [int(s // patch_size) for s in _img_size]
    _img_crop_size = [s * patch_size for s in _feat_map_size]
    x = x[:, : _img_crop_size[0], : _img_crop_size[1]]
    x_dict = dino_model.forward_features(x.unsqueeze(0))
    x_feat_map = (
        x_dict["x_norm_patchtokens"][0]
        .reshape(
            *(
                _feat_map_size
                + [
                    -1,
                ]
            )
        )
        .permute(2, 0, 1)
    )

    if seg_mask is not None:
        mask = torch.from_numpy(seg_mask)
        mask_crop = mask[: _img_crop_size[0], : _img_crop_size[1]]
        mask_transform = transforms.Resize(
            size=_feat_map_size, interpolation=transforms.InterpolationMode.NEAREST
        )
        mask_feat_map = mask_transform(mask_crop[None,])[0]
        frame["mask_feat_map"] = mask_feat_map

    frame["feat_map"] = x_feat_map
    frame["feat_map_np"] = frame["feat_map"].cpu().detach().numpy()
    frame["feats_np"] = (
        frame["feat_map"].flatten(1).permute(1, 0).cpu().detach().numpy()
    )
    frame["feat_patch_size"] = patch_size
    frame["image"] = image

    return frame


def nearest_neighbor(feats_A, feats_B):
    feats_A = feats_A.reshape(feats_A.shape[0], -1)
    feats_B = feats_B.reshape(feats_B.shape[0], -1)
    distances = np.linalg.norm(
        feats_A[:, :, np.newaxis] - feats_B[:, np.newaxis, :], axis=0
    )
    match_A_to_B = np.argmin(distances, axis=1)
    match_A_to_B_scores = -np.min(distances, axis=1)
    return match_A_to_B, match_A_to_B_scores


def idx_to_xy(idx, map2d, scale=None):
    W = map2d.shape[-1]
    if idx is None:
        idx = np.arange(map2d.shape[-1] * map2d.shape[-2])
    y = idx // W
    x = idx % W
    if scale is not None:
        y = (y + 0.5) * scale
        x = (x + 0.5) * scale
    xy = np.stack([x, y], axis=-1)
    return xy


def get_matches(image_a, image_b, seg_a=None, seg_b=None, mask=True):
    start_time_forward = time.time()
    dict_a = forward(image_a, seg_a)
    dict_b = forward(image_b, seg_b)
    end_time_forward = time.time()

    feats_A = dict_a["feat_map_np"]
    feats_B = dict_b["feat_map_np"]

    start_time_pca = time.time()
    if pca_components > 0:
        feats = np.concatenate([dict_a["feats_np"], dict_b["feats_np"]], axis=0)
        pca = PCA(n_components=pca_components)
        pca.fit(feats)
        feats_A = (
            pca.transform(dict_a["feats_np"])
            .reshape(feats_A.shape[1], feats_A.shape[2], pca_components)
            .transpose((2, 0, 1))
        )
        feats_B = (
            pca.transform(dict_b["feats_np"])
            .reshape(feats_B.shape[1], feats_B.shape[2], pca_components)
            .transpose((2, 0, 1))
        )
    end_time_pca = time.time()

    start_time_knn = time.time()
    match_A_to_B, match_A_to_B_scores = nearest_neighbor(feats_A, feats_B)
    end_time_knn = time.time()

    xy_A = idx_to_xy(None, dict_a["feat_map"])

    if mask:
        match_A_to_B_scores[~dict_a["mask_feat_map"][xy_A[:, 1], xy_A[:, 0]]] = -np.inf

    print(f"Matches found: {np.sum(match_A_to_B_scores != -np.inf)}")
    # print(f"Forward time: {end_time_forward - start_time_forward}")
    # print(f"PCA time: {end_time_pca - start_time_pca}")
    # print(f"KNN time: {end_time_knn - start_time_knn}")
    return dict_a, dict_b, match_A_to_B, match_A_to_B_scores


def display_matches(frame_A, frame_B, points_A, points_B):
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    fig = plt.figure(figsize=(10, 8))
    ax_A = fig.add_subplot(121)
    ax_B = fig.add_subplot(122)

    ax_A.imshow(frame_A["image"])
    ax_B.imshow(frame_B["image"])

    for xy_A, xy_B in zip(points_A, points_B):
        con = ConnectionPatch(
            xyA=xy_A,
            xyB=xy_B,
            coordsA="data",
            coordsB="data",
            axesA=ax_A,
            axesB=ax_B,
            color=np.random.rand(
                3,
            ),
        )
        ax_B.add_artist(con)
    plt.show()


def get_dino_keypoints_in_mask(
    image_a, image_b, seg_a, debug_vis: bool = False, clean_up: bool = False
):
    frame_A, frame_B, match_A_to_B, match_A_to_B_scores = get_matches(
        image_a, image_b, seg_a, seg_b=None
    )
    points_a = []
    points_b = []
    if match_A_to_B_scores is not None:
        match_ids_A = np.argsort(match_A_to_B_scores)[::-1]
    else:
        match_ids_A = np.random.permutation(len(match_A_to_B_scores))

    for id_A in match_ids_A:
        if match_A_to_B_scores[id_A] == -np.inf:
            continue

        id_B = match_A_to_B[id_A]
        xy_A = idx_to_xy(id_A, frame_A["feat_map"], scale=frame_A["feat_patch_size"])
        xy_B = idx_to_xy(id_B, frame_B["feat_map"], scale=frame_B["feat_patch_size"])

        points_a.append(xy_A)
        points_b.append(xy_B)
        confidences = np.ones(len(points_a))

    if debug_vis:
        display_matches(frame_A, frame_B, points_a, points_b)

    return (
        np.array(points_a).astype(int)[..., ::-1],
        np.array(points_b).astype(int)[..., ::-1],
        confidences,
    )
