def detection_collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

