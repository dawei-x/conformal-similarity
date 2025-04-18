import os
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import json
from tqdm import tqdm

def load_json_file(json_file):
    with open(json_file, 'r') as f:
        json_loaded = json.load(f)
    return json_loaded

def get_representative_by_centroid(extractor, image_folder, label, images_per_class=50):
    start_idx = label * images_per_class
    end_idx = start_idx + images_per_class

    images = [image_folder[i][0] for i in range(start_idx, end_idx)]
    batch = torch.stack(images).cuda()  # shape: [50, 3, 224, 224]

    with torch.no_grad():
        preds = extractor(batch)        # [50, D, 1, 1]
        preds = torch.flatten(preds, 1)         # [50, D]
        preds = F.normalize(preds, dim=1)       # cosine-normalized [50, D]

        centroid = F.normalize(preds.mean(dim=0), dim=0)  # [D]
        similarities = preds @ centroid                   # [50]
        best_local_idx = similarities.argmax().item()

    best_global_idx = start_idx + best_local_idx
    best_embedding = preds[best_local_idx]

    return best_global_idx, best_embedding

def get_class_proxies(extractor, image_folder, num_classes = 1000, images_per_class = 50):

    representatives = {}

    with torch.no_grad():
        for label in tqdm(range(num_classes)):
            idx, emb = get_representative_by_centroid(extractor, image_folder, label, images_per_class)
            representatives[label] = {'image_index': idx, 'embedding': emb}

    return representatives


def select_stimuli(model, data_loader, indices, min_size = 2, max_size = 10):
    stimuli_list = []
    with torch.no_grad():
        start = 0  # batch start index
        for imgs, labels in tqdm(data_loader):
            _, pred_sets = model(imgs.cuda())
            for i in range(len(pred_sets)):
                pred_set = pred_sets[i].tolist()
                if min_size <= len(pred_set) <= max_size:
                    file_index = indices[start + i]
                    stimuli_list.append({'file_index': file_index, 'set_size': len(pred_set), 'prediction_set': pred_set})

            start += imgs.shape[0]
    return stimuli_list


def compute_set_similarity(pred_set, proxy_lookup):
    embeddings = []

    # for prediction sets that contain zero or only one label.
    if len(pred_set) < 2:
        return 1.0, 1.0, 1.0

    embeddings = [proxy_lookup[label]['embedding'] for label in pred_set]

    embeddings = torch.stack(embeddings)

    sim_matrix = embeddings @ embeddings.T

    num = sim_matrix.size(0)
    sim_matrix.fill_diagonal_(0)

    triu_indices = torch.triu_indices(num, num, offset=1)
    sim_values = sim_matrix[triu_indices[0], triu_indices[1]]

    average_similarity = sim_values.mean().item()
    median_similarity = sim_values.median().item()
    min_similarity = sim_values.min().item()

    return average_similarity, median_similarity, min_similarity

def curate_proxy_images(instance_index, pred_set, proxy_lookup, label_map, image_folder, target_dir):
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
        )

    # save the stimulus image
    instance_img, instance_label = image_folder[instance_index]
    instance_name = label_map[instance_label]
    img_inv = inv_normalize(instance_img)
    img_pil = TF.to_pil_image(img_inv.clamp(0, 1))  # Clamp to valid range
    instance_filename = f"(INSTANCE) {instance_name}_{instance_index}.jpg"
    img_path = os.path.join(target_dir, instance_filename)
    img_pil.save(img_path)

    # save the proxy image for each label in the prediction set
    for label in pred_set:
        proxy_idx = proxy_lookup[label]['image_index']
        img, _ = image_folder[proxy_idx]
        labelname = label_map[label]
        img_inv = inv_normalize(img)
        img_pil = TF.to_pil_image(img_inv.clamp(0, 1))
        filename = f"{labelname}.jpg"
        img_path = os.path.join(target_dir, filename)
        img_pil.save(img_path)