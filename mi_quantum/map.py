import torch
import torch.nn as nn
from numpy import np
import matplotlib.pyplot as plt

def save_attention(output,image):
    attentions = output.attentions[-1] # we are only interested in the attention maps of the last layer
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    threshold = 0.6
    w_featmap = image.shape[-2] // 14
    h_featmap = image.shape[-1] // 14 #model.config.patch_size

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=14, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=14, mode="nearest")[0].cpu()
    attentions = attentions.detach().numpy()

    # show attentions heatmaps
    image = np.transpose(image[0], (1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min())
    for j in range(nh):
        # Crear una figura
        fig, ax = plt.subplots()

        # Mostrar la imagen de fondo
        ax.imshow(image)

        # Superponer la imagen con transparencia alpha
        attentions_normalized = (attentions[j] - attentions[j].min())/ (attentions[j].max()-attentions[j].min())
        ax.imshow(attentions_normalized, alpha=0.5)

        # Ocultar los ejes
        ax.axis('off')

        # Guardar la imagen resultante
        plt.savefig("./derma_results/_attn-head_"+str(j)+".jpg", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    return
