import torch
from torch import nn
from mi_quantum.quantum.pennylane_backend import QuantumLayer
import numbers

# See:
# - https://nlp.seas.harvard.edu/annotated-transformer/
# - https://github.com/rdisipio/qtransformer/blob/main/qtransformer.py
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

class MultiheadSelfAttention(nn.Module):
        def __init__(self, embed_dim, num_heads, dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}, RBF_similarity = False, tensor_dimension = 2):
            super().__init__()
            assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.tensor_dimension = tensor_dimension
            assert RBF_similarity == 'none' or RBF_similarity == 'quantum' or RBF_similarity == 'linear' or (isinstance(RBF_similarity, numbers.Real) and 0 < RBF_similarity <= 1), f"RBF_similarity must be set to 'none', 'quantum', 'linear' or a float between 0 and 1, but got {RBF_similarity}"
            self.RBF_similarity = RBF_similarity
            print(f'Setting up multihead self-attention with RBF_similarity: {self.RBF_similarity}')

            if self.RBF_similarity == 'quantum':
                self.quantum_ponderation = QuantumLayer(num_qubits=2, entangle= True, trainBool= True, graph= 'chain' )
            elif self.RBF_similarity == 'linear':
                self.ponderation_param = nn.Parameter(torch.tensor(0.0))  # initialized at 0 → sigmoid = 0.5
    
            print('Started a MutliheadSelfAttention layer with embed_dim:', embed_dim, 'num_heads:', num_heads, 'head_dim:', self.head_dim, 'RBF_similarity:', self.RBF_similarity)

            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.dropout = nn.Dropout(dropout['embedding_attn'])
            self.o_proj = nn.Linear(embed_dim, embed_dim)

        def rank_patches_by_attention(attn: torch.Tensor) -> torch.Tensor:
            """
            Ranks image patches by the total attention they receive.

            """
            # Average over heads: (B, T, T)
            attn_mean = attn.mean(dim=1)

            # Total attention received by each token: sum over the source positions (axis=-2)
            # attention_received[b, j] = sum over i of attn[b, i, j]
            attention_received = attn_mean.sum(dim=1)  # shape: (B, T)

            # Sort patches by total attention received, descending
            sorted_indices = attention_received.argsort(dim=1, descending=True)  # shape: (B, T)

            return sorted_indices



        def forward(self, x):
            batch_size, seq_len, embed_dim = x.shape
            # x.shape = (batch_size, seq_len, embed_dim)
            assert embed_dim == self.embed_dim, f"Input embedding dimension ({embed_dim}) should match layer embedding dimension ({self.embed_dim})"

            q, k, v = [
                proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                for proj, x in zip([self.q_proj, self.k_proj, self.v_proj], [x, x, x])
            ]

            # Compute scaled dot-product attention
            qk_dot = q @ k.transpose(-2, -1)

            attn_logits = ( qk_dot / (self.head_dim ** 0.5))
            # attn_logits.shape = (batch_size, num_heads, seq_len, seq_len)
            attn = attn_logits.softmax(dim=-1)
            # attn.shape = (batch_size, num_heads, seq_len, seq_len)
            attn = self.dropout(attn)

            if self.RBF_similarity != 'none':
                if self.tensor_dimension != 2:
                    raise ValueError("RBF similarity is only implemented for standard 2D tensors products (Query and Key) ")
                # Compute RBF similarity
                if self.RBF_similarity == 'quantum':
                    ponderators = torch.sigmoid( self.quantum_ponderation( torch.zeros( (2,), device = x.device) ))
                elif isinstance(self.RBF_similarity, numbers.Real):    
                    ponderators = torch.tensor([1 - self.RBF_similarity, self.RBF_similarity], device=x.device, dtype=torch.float32)
                else: # RBF_similarity == 'linear'
                    sigmoid_val = torch.sigmoid(self.ponderation_param)
                    ponderators = torch.stack([1 - sigmoid_val, sigmoid_val])

                # promote to float32 for stability
                q32 = q.float()
                k32 = k.float()

                q_norm = (q32 ** 2).sum(dim=-1, keepdim=True)            # (B,H,S,1)
                k_norm = (k32 ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)  # (B,H,1,S)

                # distance
                dists_squared = q_norm + k_norm - 2 * (q32 @ k32.transpose(-2, -1))

                # clamp negative numerical noise
                dists_squared = dists_squared.clamp_min(0.0)

                # scale (avoid too-small / too-large)
                sigma_squared = q_norm.clamp_min(1e-8).clamp_max(1e4)  # tune upper bound as needed

                attn_RBF_logits = torch.exp(-dists_squared / sigma_squared)

                denom = attn_RBF_logits.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                attn_RBF = attn_RBF_logits / denom

                attn_RBF = self.dropout(attn_RBF)

                if (~torch.isfinite(attn_RBF)).any():
                    import warnings
                    print(f"Registered non-finite attn. ponderators={ponderators}")
                    warnings.warn(f"Non-finite RBF attention detected. Ponderators = {ponderators}", RuntimeWarning)
                    
                # sanitize just in case
                attn_RBF = torch.nan_to_num(attn_RBF, nan=0.0, posinf=0.0, neginf=0.0)

                # mix (ponderators assumed roughly sum=1 but be safe)

                weight_sum = ponderators[0] + ponderators[1]
                attn = (ponderators[0] * attn + ponderators[1] * attn_RBF) / (weight_sum + 1e-7)


            # Compute output
            values = attn @ v
            # values.shape = (batch_size, num_heads, seq_len, head_dim)
            values = values.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
            # values.shape = (batch_size, seq_len, embed_dim)
            x = self.o_proj(values)
            # x.shape = (batch_size, seq_len, embed_dim)

            return x, attn

class FeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size, hidden_size_out , quantum = True, dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
                    q_stride = 4, trainBool = False, entangle = True, graph = 'chain'):
        super().__init__()

        self.quantum = quantum
        self.trainBool = trainBool
        self.q_stride = q_stride
        self.mlp_hidden_size = mlp_hidden_size

        self.fc1 = nn.Linear(hidden_size, q_stride * mlp_hidden_size)
        self.fc2 = nn.Linear(q_stride * mlp_hidden_size, hidden_size_out)

        if self.quantum:
            self.vqc = QuantumLayer(mlp_hidden_size, entangle = entangle, trainBool = self.trainBool, graph = graph)
        else:
            self.vqc = nn.Linear(mlp_hidden_size, mlp_hidden_size)

        self.dropout = nn.Dropout(dropout['feedforward'])
        self.gelu = nn.GELU()
        self.q_stride = q_stride
        

    def forward(self, x):
        device = x.device

        if self.q_stride == 1:
            x = self.fc1(x)
            x = self.vqc(x)
            x = x.to(device)  # Ensure the output is on the same device as the input
            x = self.dropout(x)
            x = self.gelu(x)
            x = self.fc2(x)
        else:
            
            x = self.fc1(x)  # x shape: [B, C, L]

            # Extract q_stride slices of size mlp_hidden_size
            slices = [x[:, :, i : i + self.mlp_hidden_size] for i in range(self.q_stride)]

            # Stack into a single batch: shape [q_stride, B, C, mlp_hidden_size]
            x_slices = torch.stack(slices, dim=0)

            # Merge batch for parallel processing: [q_stride * B * C, mlp_hidden_size]
            q, B, C, H = x_slices.shape
            x_slices = x_slices.permute(1, 2, 0, 3).contiguous().view(-1, H)

            # Apply vqc in batch
            x_vqc_output = self.vqc(x_slices)  # returns [self.q_stride * B * C, D]

            # Reshape back: [B, C, q_stride, D]
            D = x_vqc_output.shape[-1]
            x_vqc_output = x_vqc_output.view(B, C, self.q_stride, D)

            # Concatenate outputs along last dimension
            x = x_vqc_output.permute(0, 1, 3, 2).contiguous().view(B, C, -1)

            # Continue forward
            x = self.dropout(x)
            x = self.gelu(x)
            x = self.fc2(x)


        return x

class TransformerBlock_Attention_Chosen_QMLP(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_hidden_size, hidden_size_out, quantum_mlp = True, RBF_similarity= False, dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}, 
                    attention_selection="filter", train_q = False, entangle = True, q_stride = 4, connectivity = 'chain', RD = 1, img_size = 28, patch_size = 4):
        super().__init__()

        self.attention_selection = attention_selection
        self.quantum_mlp = quantum_mlp
        self.train_q = train_q
        self.RBF_similarity = RBF_similarity
        self.dropout = dropout
        # Attention components
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = MultiheadSelfAttention(hidden_size, num_heads, dropout, self.RBF_similarity)
        self.attn_dropout = nn.Dropout(dropout['after_attn'])
        self.hidden_size_out = hidden_size_out
        self.RD = RD

        # MLP components
        self.mlp_norm = nn.LayerNorm(hidden_size)

        self.mlp_sel = FeedForward(hidden_size, mlp_hidden_size, hidden_size_out, quantum = self.quantum_mlp,
                                    dropout = self.dropout, trainBool = self.train_q, 
                                    entangle = entangle, q_stride = q_stride, graph = connectivity)  # Quantum MLP

        if attention_selection != "filter" or RD > 1:
            self.mlp = nn.Linear(hidden_size, hidden_size_out) if attention_selection != "ID" else nn.Identity()

        self.q_lr = (img_size * mlp_hidden_size) // patch_size  # Number of high-attention patches to select
        self.mlp_dropout = nn.Dropout(dropout['feedforward'])

        if attention_selection == "ID" and hidden_size != hidden_size_out:
            raise ValueError("When attention_selection is 'ID', hidden_size must equal hidden_size_out.")

    def forward(self, x):
        # Attention block
        attn_input = self.attn_norm(x)
        attn_output, attn_map = self.attn(attn_input)
        attn_output = self.attn_dropout(attn_output)
        x = x + attn_output
        y = self.mlp_norm(x)

        # MLP input
        if self.attention_selection != "none":

            # Rank patches by attention
            attn_indices = MultiheadSelfAttention.rank_patches_by_attention(attn_map)
            sel_indices = attn_indices[:, :self.q_lr]       # High-attention patches
            normal_indices = attn_indices[:, self.q_lr:]      # Remaining patches

            # Ensure CLS token is always included
            cls_index = torch.zeros(sel_indices.size(0), 1, dtype=torch.long, device=sel_indices.device)
            sel_indices = torch.cat([cls_index, sel_indices[:, :-1]], dim=1)

            # Feedforward on selected patches
            y_sel_in = y.gather(1, sel_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            y_sel_out = self.mlp_sel(y_sel_in)

            # Classical MLP on the rest. Note that if quantum is False, then this is sort of redundant.
            if self.attention_selection == "MLP":
                y_normal_in = y.gather(1, normal_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
                y_normal_out = self.mlp(y_normal_in)
                y_normal_out = self.mlp_dropout(y_normal_out)

            elif self.attention_selection == "ID":
                y_normal_out = y.gather(1, normal_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))

            elif self.attention_selection == "filter":
                # If attention_selection is 'filter', we do not apply a classical MLP
                y_out = y_sel_out
                x = x.gather(1, sel_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))) if self.RD == 1 else self.mlp(x.gather(1, sel_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))))
                return x + y_out, attn_map

            else:
                raise ValueError(f"Unknown attention_selection: {self.attention_selection}")

            # Combine and return, preserving original order
            batch_size, num_tokens, dim = x.size()
            device = x.device

            # Create empty tensor to hold ordered outputs
            y_out = torch.zeros((batch_size, num_tokens, self.hidden_size_out), device=device)

            # Place outputs back in their original positions
            y_out.scatter_(1, sel_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size_out), y_sel_out)
            y_out.scatter_(1, normal_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size_out), y_normal_out)

            return x + y_out, attn_map

        else:
            # If no attention selection, use standard MLP
            y = self.mlp(y)
            y = self.mlp_dropout(y)
            return x + y, attn_map


class VisionTransformer(nn.Module):
    def __init__(self, img_size, num_channels, num_classes, patch_size, hidden_size, num_heads, num_transformer_blocks, mlp_hidden_size, 
                    quantum_mlp = False, RBF_similarity = 'none', quantum_classification = False, dropout= {'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}, 
                    channels_last=False, RD = 1, attention_selection = 'filter', 
                    paralel = 1, train_q = False, entangle = True, q_stride = 1, connectivity = 'chain'
                    ):
        super().__init__()

        self.trainlosslist = []
        self.trauclist = []
        self.tracclist = []
        self.vallosslist = []
        self.auclist = []
        self.acclist = []
        self.attention_maps = []

        self.channels_last = channels_last
        self.RD = RD
        self.paralel = paralel
        self.num_transformer_blocks = num_transformer_blocks
        
        self.attention_selection = attention_selection
        self.starting_dim = num_channels * patch_size ** 2
        self.dropout_values = dropout

        self.quantum_mlp = quantum_mlp
        self.quantum_classification = quantum_classification
        self.train_q = train_q
        self.RBF_similarity = RBF_similarity

        self.train_q = train_q
        self.entangle = entangle
        self.q_stride = q_stride
        self.connectivity = connectivity

        # Splitting an image into patches and linearly projecting these flattened patches can be
        # simplified as a single convolution operation, where both the kernel size and the stride size
        # are set to the patch size.
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (img_size // patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        num_steps = 1 + num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, hidden_size) * 0.02)
        self.dropout = nn.Dropout(self.dropout_values['embedding_pos'])

        
        # Transformer blocks with attention selection
        self.transformer_blocks = nn.ModuleList( [nn.ModuleList([TransformerBlock_Attention_Chosen_QMLP(hidden_size // RD**i, num_heads, mlp_hidden_size, hidden_size // RD**(i + 1) , quantum_mlp = self.quantum_mlp,
                                                                                        RBF_similarity= self.RBF_similarity ,dropout = self.dropout_values,
                                                                                        attention_selection = self.attention_selection,
                                                                                        train_q = self.train_q, entangle = self.entangle, q_stride = self.q_stride, connectivity = self.connectivity)
                                            for i in range(num_transformer_blocks)]) for j in range(paralel) ] )

        self.layer_norm = nn.LayerNorm(hidden_size // (RD**(num_transformer_blocks)))  # Normalization after the last transformer block

        self.linear = nn.Linear( (hidden_size // (RD**(num_transformer_blocks)) ) * paralel, num_classes)
        self.linear2 = nn.Linear(num_classes,num_classes) if not self.quantum_classification else QuantumLayer(num_qubits=num_classes, entangle=self.entangle, trainBool= self.train_q, graph=self.connectivity)

        

    def forward(self, x):
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        # x.shape = (batch_size, num_channels, img_size, img_size)

        x = self.patch_embedding(x)
        # x.shape = (batch_size, hidden_size, sqrt(num_patches), sqrt(num_patches))
        x = x.flatten(start_dim=2)
        # x.shape = (batch_size, hidden_size, num_patches)
        x = x.transpose(1, 2)
        # x.shape = (batch_size, num_patches, hidden_size)

        # CLS token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x.shape = (batch_size, num_steps, hidden_size)

        # Positional embedding
        x = self.dropout(x + self.pos_embedding)  # [B, S, D]

        # Repeat x for each parallel branch
        x_parallel = x.unsqueeze(0).repeat(self.paralel, 1, 1, 1)  # [P, B, S, D]

        attn_maps = []
        outputs = []

        for i in range(self.paralel):
            out = x_parallel[i]  # [B, S, D]
            for j in range(self.num_transformer_blocks):
                out, attn = self.transformer_blocks[i][j](out)  # [B, S, D], attn: [B, H, S, S] or similar
                attn_maps.append(attn)

            out = self.layer_norm(out)         # [B, S, D]
            out = out[:, 0]                    # [B, D]
            outputs.append(out)                # Collect [B, D]

        # Concatenate along hidden dimension
        x = torch.cat(outputs, dim=1)  # [B, D * P]


        # Classification logits
        x = self.linear(x)
        x = self.linear2(x)

        # x.shape = (batch_size, num_classes)
        return x, attn_maps
    
class Encoder(nn.Module):
                    def __init__(self, encoder_layers, dropout_pos):
                        super(Encoder, self).__init__()
                        self.encoder_layers = encoder_layers
                        self.dropout_pos = dropout_pos


                    def forward(self, x, pos_embedding):
                        # Apply patch and position embeddings, including the class token
                        
                        x += pos_embedding[:, :(x.shape[1])]
                        out = self.dropout_pos(x)
                        

                        for layer in self.encoder_layers:
                            
                            out, _ = layer(out)
    
                            if type(out) != torch.Tensor:
                                raise ValueError("The output is not a tensor.")
                            
                        if type(out) != torch.Tensor:
                            raise ValueError("The output is not a tensor.")
                        return out  # Shape: (paralel, batch_size, num_patches + 1, hidden_size)

class Decoder(nn.Module):
    def __init__(self, decoder_layers):
        super(Decoder, self).__init__()
        self.decoder_layers = decoder_layers


    def forward(self, z):
        outputs = []
        # Pass the latent representation through each decoder block
        out = z
        for layer in self.decoder_layers:
            out, _ = layer(out)

        # Concatenate outputs from all parallel branches
        return out
    
class AutoEnformer(nn.Module):
            def __init__(self, img_size, num_channels, patch_size, hidden_size, num_heads, num_transformer_blocks, RBF_similarity ,mlp_hidden_size,
                              dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}, channels_last=False, attention_selection='none', RD=1,
                              q_stride = 1):
                super(AutoEnformer, self).__init__()

                self.channels_last = channels_last
                self.RD = RD
                self.trainlosslist = []
                self.vallosslist = []
                
                self.attention_selection = attention_selection
                self.starting_dim = num_channels * patch_size ** 2
                self.dropout_values = dropout
                self.num_channels = num_channels

                self.RBF_similarity = RBF_similarity

                self.q_stride = q_stride

                self.patch_embedding = nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=hidden_size,
                    kernel_size=patch_size,
                    stride=patch_size
                )
                num_patches = (img_size // patch_size)**2

                num_steps = 1 + num_patches

                self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, hidden_size) * 0.02)
                self.dropout = nn.Dropout(self.dropout_values['embedding_pos'])

                self.encoder_layers = nn.ModuleList([TransformerBlock_Attention_Chosen_QMLP(hidden_size // RD**i, num_heads, mlp_hidden_size, hidden_size // RD**(i + 1) , quantum_mlp = self.quantum_mlp,
                                                                                                    RBF_similarity= self.RBF_similarity ,dropout = self.dropout_values,
                                                                                                    attention_selection = 'none',
                                                                                                    train_q = False, entangle = False, q_stride = self.q_stride )
                                                        for i in range(num_transformer_blocks)]) 
                
                self.decoder_layers = nn.ModuleList([TransformerBlock_Attention_Chosen_QMLP(hidden_size // RD**i, num_heads, mlp_hidden_size, hidden_size // RD**(i + 1) , quantum_mlp = self.quantum_mlp,
                                                                                            RBF_similarity= self.RBF_similarity ,dropout = self.dropout_values,
                                                                                            attention_selection = 'none',
                                                                                            train_q = False, entangle = False, q_stride = self.q_stride )
                                                        for i in range(num_transformer_blocks, 0, -1) ]) 
                                        
                self.Encoder = Encoder(self.encoder_layers, self.dropout)
                self.Decoder = Decoder(self.decoder_layers)

            def get_latent_representation(self, img):

                # Check and handle channel-last format if needed
                if self.channels_last:
                    img = img.permute(0, 3, 1, 2)
                
                # Create image patches and project them to the hidden size
                x = self.patch_embedding(img)
                x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N is number of patches

                # Apply patch and position embeddings, including the class token
                x += self.pos_embedding[:, :(x.shape[1])]
                out = self.dropout(x)
                
                for layer in self.encoder_layers[:-1]: # Exclude the last layer to get latent representations
                    out, _ = layer(out)

                last_layer = self.encoder_layers[-1]
                attn_input = last_layer.attn_norm(out)
                attn_output, attn_map = last_layer.attn(attn_input)
                attn_output = last_layer.attn_dropout(attn_output)
                out = out + attn_output
                y = last_layer.mlp_norm(out)
                mlp_out = last_layer.mlp_sel.fc1(y)
                mlp_out = last_layer.mlp_sel.vqc(mlp_out)
                mlp_out = mlp_out.to(y.device)  # Ensure the output is on the same device as the input
                mlp_out = last_layer.mlp_sel.dropout(mlp_out)
                latent_representations = last_layer.mlp_sel.gelu(mlp_out)
                
                return latent_representations  # Shape: (batch_size, num_patches + 1, mlp_hidden_size)
                    
            
                        
            def forward(self, img):
                
                # 1. Prepare the input image
                # Check and handle channel-last format if needed
                if self.channels_last:
                    img = img.permute(0, 3, 1, 2)
                
                # Create image patches and project them to the hidden size
                x = self.patch_embedding(img)
                x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N is number of patches
                
                # 2. Instantiate and run the Encoder

                latent_representations = self.Encoder(x, self.pos_embedding)
                
                reconstructed_patches = self.Decoder(latent_representations)

                # 4. Final Reconstruction              
                
                # Transpose to get the channel dimension for reshaping
                reconstructed_patches = reconstructed_patches.transpose(1, 2)
                
                # Reshape to a 4D tensor
                reconstructed_imgs = reconstructed_patches.reshape(img.shape)

                #errors = [ torch.mean((img - recon)**2).item() for recon in reconstructed_imgs ]
                
                return reconstructed_imgs#, errors # Return the reconstructed images and the reconstruction errors
            

class DeViT(nn.Module):
            """ Vision Transformer for classification on top of latent representations.
                First, a linear layer adjusts the dimension of the latent representation to match the ViT input size.
                Then, a standard ViT is applied for classification.
            """
            def __init__(self, num_classes, p, shape):
                super(DeViT, self).__init__()

                self.num_classes = num_classes
                self.p = p
                self.shape = shape
                self.dimension_adjustment = nn.Linear(p['mlp_size'], shape[0] * p['patch_size']**2)

                self.trainlosslist = []
                self.trauclist = []
                self.tracclist = []
                self.vallosslist = []
                self.auclist = []
                self.acclist = []
                self.attention_maps = []

                self.vit = VisionTransformer(
                    img_size=28, num_channels=3, num_classes=num_classes,
                    patch_size=p['patch_size'], hidden_size=shape[0] * p['patch_size']**2, num_heads=p['num_head'],
                    num_transformer_blocks=p['num_transf'], attention_selection='none', RBF_similarity = 'none',
                    mlp_hidden_size=p['mlp_size'], quantum_mlp = False, dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}, channels_last=False, entangle=False, quantum_classification = False,
                    paralel = p['paralel'], RD = 1, train_q = False, q_stride = 1, connectivity = 'chain'
                )



            def forward(self, x):  
                x = self.dimension_adjustment(x)  
                x = x.reshape((x.shape[0],) + self.shape)
                return self.vit(x)

