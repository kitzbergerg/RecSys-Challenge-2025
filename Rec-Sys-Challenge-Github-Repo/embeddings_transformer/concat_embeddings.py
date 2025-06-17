import numpy as np

client_ids = np.load("../results/transformer/v2/client_ids.npy")
embeddings = np.load("../results/transformer/v2/embeddings.npy")

client_ids_other = np.load("../results/embeddings_deep_autoencoder/client_ids.npy")
embeddings_other = np.load("../results/embeddings_deep_autoencoder/embeddings.npy")

print(client_ids.shape)
print(embeddings.shape)
print(client_ids_other.shape)
print(embeddings_other.shape)

other_dict = {cid: emb for cid, emb in zip(client_ids_other, embeddings_other)}

client_ids_concat = []
embeddings_concat = []

for cid, emb in zip(client_ids, embeddings):
    if cid in other_dict:
        client_ids_concat.append(cid)
        embeddings_concat.append(np.concat((emb, other_dict[cid])))
    else:
        print("WARNING: client_id {} not in other_dict".format(cid))
        exit(0)

client_ids_concat = np.array(client_ids_concat)
embeddings_concat = np.stack(embeddings_concat)

print(client_ids_concat.shape)
print(embeddings_concat.shape)
assert client_ids.shape == client_ids_concat.shape
assert embeddings.shape[0] == embeddings_concat.shape[0]
assert client_ids_concat.shape[0] == embeddings_concat.shape[0]
assert embeddings.shape[1] + embeddings_other.shape[1] == embeddings_concat.shape[1]

# Save the aligned client_ids and combined embeddings
np.save("../results/transformer_concat/client_ids.npy", client_ids_concat)
np.save("../results/transformer_concat/embeddings.npy", embeddings_concat)
