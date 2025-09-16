import torch
from tqdm import tqdm


def topk_prototype_images(model, dataloader, k=5):
    model.eval()

    test_iter = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Top-K Prototype Images",
        mininterval=2.0,
        ncols=0,
    )

    total_similarity_scores = torch.empty(0).cuda()

    for i, (inputs, _) in test_iter:
        inputs = inputs.cuda()

        with torch.no_grad():
            _, similarity_maps, _ = model.explain(inputs)

            # get the maximum value of the similarity maps
            similarity_scores = torch.functional.F.max_pool2d(
                similarity_maps, kernel_size=similarity_maps.shape[2:]
            ).squeeze()

            total_similarity_scores = torch.cat((total_similarity_scores, similarity_scores), dim=0)

        del inputs, similarity_maps

    # get the top-k similarity scores for each prototype
    # similarity scores are NxP where N is the image index and P is the number of prototypes
    _, topk_images = torch.topk(total_similarity_scores, k, dim=0)

    # make a dictionary of the top-k images for each prototype
    topk_images_dict = {}
    for i in range(topk_images.shape[1]):
        topk_ids = topk_images[:, i].cpu().tolist()
        proto_info_dict = {
            "ids": [dataloader.dataset.img_ids[idx] for idx in topk_ids],
            "paths": [dataloader.dataset.paths[idx] for idx in topk_ids],
            "labels": (
                [dataloader.dataset.labels[idx][0] for idx in topk_ids]
                if not model.multi_label
                else [dataloader.dataset.labels[idx] for idx in topk_ids]
            ),
        }
        topk_images_dict[i] = proto_info_dict

    return topk_images_dict
