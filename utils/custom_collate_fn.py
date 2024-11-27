import torch

def custom_collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    sequences, labels = zip(*batch)
    max_len = max([s.shape[0] for s in sequences])
    padded_seqs = []
    for seq in sequences:
        seq_len = seq.shape[0]
        if seq_len < max_len:
            last_frame = seq[-1].unsqueeze(0)
            num_repeat = max_len - seq_len
            padding = last_frame.repeat(num_repeat, 1, 1, 1)
            padded = torch.cat([seq, padding], dim=0)
        else:
            padded = seq
        padded_seqs.append(padded)
    padded_seqs = torch.stack(padded_seqs, dim=0)

    labels = torch.stack(labels)
    return padded_seqs, labels