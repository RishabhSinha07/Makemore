"""makemore: a character-level bigram language model for generating names.

Extracted from build_makemore_yay.ipynb. Walks through two approaches:
  1. A count-based bigram model (build a probability matrix from char pairs).
  2. The same model recast as a single-layer neural network trained with
     gradient descent.

Dataset: datasets/names/{female,male}.txt
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
def clean(dataset):
    dataset = [name.strip().lower() for name in dataset]
    dataset = [''.join(filter(str.isalpha, name)) for name in dataset]
    return dataset


ds_fnames = open("datasets/names/female.txt").read().splitlines()
ds_mnames = open("datasets/names/male.txt").read().splitlines()

print("Examples:", ds_fnames[:10], "\nTotal:", len(set(ds_fnames)))
print("Min:", min(len(l) for l in ds_fnames), "Max:", max(len(l) for l in ds_fnames))

print("Examples:", ds_mnames[:10], "\nTotal:", len(set(ds_mnames)))
print("Min:", min(len(l) for l in ds_mnames), "Max:", max(len(l) for l in ds_mnames))

training_ds = clean(ds_fnames) + clean(ds_mnames)


# ---------------------------------------------------------------------------
# 2. Build the bigram count matrix
#
# A bigram maps a character to the next character. Names are wrapped with "."
# as a single start/end marker. Using one symbol for both (rather than separate
# ^ and $) avoids empty rows/cols in the matrix: ^ never follows a char and $
# never precedes one.
# ---------------------------------------------------------------------------
count = {}

for name in training_ds:
    name = "." + name + "."
    for x, y in zip(name, name[1:]):
        count[(x, y)] = count.get((x, y), 0) + 1

unique_chars = sorted(list(set("".join([k[0] + k[1] for k in count.keys()]))))

# Initialised to ones for add-one (Laplace) smoothing: avoids zero counts that
# would make log-likelihood blow up. The "not_trained" matrix stays uniform as
# a baseline for comparison.
biagram_probability = np.ones((len(unique_chars), len(unique_chars)), dtype=np.int32)
biagram_probability_not_trained = np.ones((len(unique_chars), len(unique_chars)), dtype=np.int32)

for (x, y), c in count.items():
    i = unique_chars.index(x)
    j = unique_chars.index(y)
    biagram_probability[i, j] += c


# ---------------------------------------------------------------------------
# 3. Visualisation helper
# ---------------------------------------------------------------------------
def viz_bigram(arr, labels=unique_chars, figsize=(16, 16), cmap="Blues"):
    """Visualize a bigram count/prob matrix as an annotated heatmap.

    Each cell shows the character pair (row -> col) and its value, with the
    background shaded by magnitude. Rows are the "from" char, cols the "to" char.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr, cmap=cmap)

    # Use integer formatting for counts, otherwise show 2 decimals (e.g. probs)
    is_int = np.issubdtype(arr.dtype, np.integer)

    for i in range(n):
        for j in range(n):
            pair = labels[i] + labels[j]
            val = arr[i, j]
            ax.text(j, i, pair, ha="center", va="bottom", color="gray", fontsize=8)
            txt = f"{val}" if is_int else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="top", color="gray", fontsize=8)

    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("next char")
    ax.set_ylabel("current char")
    ax.axis("off")
    plt.show()


# viz_bigram(biagram_probability)
# viz_bigram(biagram_probability_not_trained)


# ---------------------------------------------------------------------------
# 4. Sample a single character from the start row
# ---------------------------------------------------------------------------
target_row = biagram_probability[0]          # row for the start char "."
target_row = target_row / target_row.sum()   # normalise to a distribution

print(target_row)
print(target_row.sum())   # Should be 1.0
print(target_row.shape)
print(target_row[0].dtype)

sample_idx = torch.multinomial(torch.tensor(target_row), num_samples=1).item()
print(unique_chars[sample_idx])


# ---------------------------------------------------------------------------
# 5. Generate names by sampling the bigram matrix until "." is drawn
# ---------------------------------------------------------------------------
def generate_from_matrix(matrix, n=10):
    names = []
    for _ in range(n):
        name = "."
        while True:
            row_idx = unique_chars.index(name[-1])
            row = matrix[row_idx]
            row = row / row.sum()  # normalise to probabilities
            sample_idx = torch.multinomial(torch.tensor(row), num_samples=1).item()
            next_char = unique_chars[sample_idx]
            name += next_char
            if next_char == ".":
                break
        names.append(name[1:-1])  # strip start/end markers
    return names


print("Generated Names using trained bigram probability matrix:")
print("\n".join(generate_from_matrix(biagram_probability)))

print("\nGenerated Names using un-trained bigram probability matrix:")
print("\n".join(generate_from_matrix(biagram_probability_not_trained)))


# ---------------------------------------------------------------------------
# 6. Pre-normalise once instead of per-step
# ---------------------------------------------------------------------------
normalized_biagram_probability = biagram_probability / biagram_probability.sum(axis=1, keepdims=True)
# viz_bigram(normalized_biagram_probability, cmap="Reds")

for _ in range(10):
    name = "."
    while True:
        row_idx = unique_chars.index(name[-1])
        row = normalized_biagram_probability[row_idx]
        sample_idx = torch.multinomial(torch.tensor(row), num_samples=1).item()
        next_char = unique_chars[sample_idx]
        name += next_char
        if next_char == ".":
            break
    print(name[1:-1])


# ---------------------------------------------------------------------------
# 7. Model evaluation: average negative log-likelihood (NLL)
#
# Lower is better. Compares the trained matrix against the uniform baseline.
# ---------------------------------------------------------------------------
def compute_nll(training_ds, unique_chars, normalized_biagram_probability):
    logProb = 0.0
    sampleSize = 0

    for name in training_ds:
        name = "." + name + "."
        for x, y in zip(name, name[1:]):
            i = unique_chars.index(x)
            j = unique_chars.index(y)
            current_prob = normalized_biagram_probability[i, j]
            logProb += torch.log(torch.tensor(current_prob))
            sampleSize += 1

    nll = -logProb / sampleSize
    print("Negative Log Likelihood (NLL) of the training dataset:", nll.item())


compute_nll(training_ds, unique_chars, normalized_biagram_probability)
compute_nll(
    training_ds,
    unique_chars,
    biagram_probability_not_trained / biagram_probability_not_trained.sum(axis=1, keepdims=True),
)


# ---------------------------------------------------------------------------
# 8. Recast as a neural network
#
# Goal: maximise the likelihood of the data w.r.t. the model parameters. Here
# the parameter is a weight matrix W trained with gradient descent, replicating
# what the count-based bigram model did by hand.
# ---------------------------------------------------------------------------
xs = []
ys = []

for name in training_ds:
    name = "." + name + "."
    for x, y in zip(name, name[1:]):
        i = unique_chars.index(x)
        j = unique_chars.index(y)
        xs.append(i)
        ys.append(j)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(xs.shape, ys.shape)


# ---------------------------------------------------------------------------
# 9. One-hot encode inputs so they can feed a matrix multiply
# ---------------------------------------------------------------------------
xs_one_hot = torch.nn.functional.one_hot(xs, num_classes=len(unique_chars)).float()
ys_one_hot = torch.nn.functional.one_hot(ys, num_classes=len(unique_chars)).float()

print(xs_one_hot.shape, ys_one_hot.shape)


# ---------------------------------------------------------------------------
# 10. Initialise weights (single linear layer, no bias)
# ---------------------------------------------------------------------------
W = torch.randn(len(unique_chars), len(unique_chars), requires_grad=True)


# ---------------------------------------------------------------------------
# 11. Training loop
#
# logits -> softmax -> NLL loss -> backprop -> gradient-descent update.
# exp(logits) plays the role of the counts; normalising gives probabilities.
# ---------------------------------------------------------------------------
EPOCHS = 100000
for k in range(EPOCHS):
    logits = xs_one_hot @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)

    loss = -probs[torch.arange(len(ys)), ys].log().mean()  # NLL loss

    W.grad = None
    loss.backward()

    W.data += -0.1 * W.grad
    if k % 1000 == 0:
        print(f"step {k}: loss {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 12. Sample names from the trained network
# ---------------------------------------------------------------------------
print("Generated Names using neural network model:")
for _ in range(10):
    name = "."
    while True:
        row_idx = unique_chars.index(name[-1])
        row = W[row_idx]
        counts = row.exp()
        probs = counts / counts.sum()
        sample_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = unique_chars[sample_idx]
        name += next_char
        if next_char == ".":
            break
    print(name[1:-1])
