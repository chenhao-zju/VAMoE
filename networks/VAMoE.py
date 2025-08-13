from functools import partial

import torch
import math
import torch.nn as nn
import torch.distributed as dist
import logging
from timm.models.layers import trunc_normal_
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .VAMoEBlock import PatchEmbed, VAMoEBlock, Mlp
from .vit_fast import Block as ViTBlock, QuickGELU, PositionEmbedder
from .l2_loss import L2_LOSS
# from VAMoEBlock import PatchEmbed, VAMoEBlock, Mlp
# from vit_fast import Block as ViTBlock, QuickGELU, PositionEmbedder
# from l2_loss import L2_LOSS


# current_rank = dist.get_rank()
# allow_print = (current_rank == 0)

class VAMoE(nn.Module):
    def __init__(
            self,
            params,
            mlp_ratio=2.,
            drop_rate=0.,
            drop_path_rate=0.,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            noisy_gating=True,
            k_element=2,
            act_layer='GELU',
            qkv_bias=True,
            window_size=(8, 8),
            use_window=True,
            interval=3,
            rel_pos_spatial=False,
            allow_print=True,
        ):
        super().__init__()
        self.params = params
        self.model_type = params['model_type']
        self.img_size = (params['h_size'], params['w_size'])
        self.patch_size = (params['patch_size'], params['patch_size'])
        self.in_chans = params['feature_dims']
        self.out_chans = params['feature_dims']
        self.num_features = self.embed_dim = params['embed_dim']
        self.num_blocks = params['num_blocks'] 
        self.depth = params['encoder_depths']
        self.use_moe = params['use_moe']
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.encoder_mlp = params['encoder_mlp']
        self.decoder_mlp = params['decoder_mlp']

        self.loss_type = params['loss']
        self.loss_weight = params['loss_weight']

        self.use_cl = params['use_cl']
        self.surface_features = params['surface_features']
        self.higher_features = params['higher_features']
        self.pressure_level =params['pressure_level']

        self.num_sf = len(self.surface_features)
        self.num_hf = len(self.higher_features)
        self.num_pl = len(self.pressure_level)

        self.topk = params['topk']


        if self.loss_type == 'trainl2':
            # print('****** using train l2 loss ******')
            learn_log_variance=dict(flag=True, channels=params['feature_dims'], logvar_init=0., requires_grad=True)
            self.loss_gen = L2_LOSS(learn_log_variance=learn_log_variance)
            self.loss_recons = L2_LOSS(learn_log_variance=learn_log_variance)

        if self.encoder_mlp:  
            mid_feature_dims = params['patch_size'] * params['patch_size'] * params['feature_dims']
            logging.info("Using MLP encoder")
        else:
            mid_feature_dims = params['embed_dim']

        # if self.use_moe=='channelmoev3':
        #     self.patch_embeds = nn.ModuleList( [PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.num_pl, embed_dim=mid_feature_dims) for _ in range(self.num_hf)] )

        #     if self.surface_features != []:
        #         self.patch_embeds.append(PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.num_sf, embed_dim=mid_feature_dims))

        #     num_patches = self.patch_embeds[0].num_patches
        # else:
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=mid_feature_dims)
        num_patches = self.patch_embed.num_patches

        if self.encoder_mlp:
            # self.encoder_mlp_layer = Mlp(in_features=mid_feature_dims, out_features=self.embed_dim, act_layer=nn.GELU, drop=drop_rate)
            self.encoder_mlp_layer = nn.Linear(mid_feature_dims, self.embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, mid_feature_dims))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]

        if self.use_moe == 'densemoe' or self.use_moe=='channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
            self.posembedder = PositionEmbedder(input_channel=self.in_chans, hidden_size=self.embed_dim)

        if self.model_type == 'VAMoE':
            if allow_print:   logging.info('********   Running VAMoE model!!!    *********')
            self.blocks = nn.ModuleList([
                VAMoEBlock(h_size=self.h, w_size=self.w, dim=self.embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction, use_moe=params['use_moe'],
                num_exports=params['num_exports'], noisy_gating=noisy_gating,
                k_element=k_element) 
            for i in range(self.depth)])

        elif self.model_type == 'vit':
            if allow_print:   logging.info('********   Running ViT model!!!    *********')
            self.blocks = nn.ModuleList([
                ViTBlock(
                    dim=self.embed_dim, num_heads=self.num_blocks, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    window_size=window_size if ((i + 1) % interval != 0) else num_patches,
                    window=((i + 1) % interval != 0) if use_window else False,
                    rel_pos_spatial=rel_pos_spatial,
                    act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU, 
                    use_moe=params['use_moe'], num_exports=params['num_exports'], 
                    noisy_gating=noisy_gating, k_element=k_element, allow_print=allow_print, topk=self.topk
                )
            for i in range(self.depth)])


        # self.norm = norm_layer(self.embed_dim)

        if self.decoder_mlp:
            # self.decoder_mlp_layer = Mlp(in_features=self.embed_dim, out_features=mid_feature_dims, act_layer=nn.GELU, drop=drop_rate)
            self.decoder_mlp_layer = nn.Linear(self.embed_dim, mid_feature_dims)

        if self.use_moe=='channelmoev3':
            self.num_interval = 4
            self.num_tiny_loss = int(self.depth/self.num_interval)

            self.tiny_loss_weights = nn.Parameter(torch.linspace(0.01, 0.1, self.num_tiny_loss))

            # print('mid_feature_dims:', mid_feature_dims)
            self.heads = nn.ModuleList([nn.Linear(mid_feature_dims, self.num_pl*self.patch_size[0]*self.patch_size[1], bias=False) for _ in range(self.num_hf)])

            if self.surface_features != []:
                self.heads.append(nn.Linear(mid_feature_dims, self.num_sf*self.patch_size[0]*self.patch_size[1], bias=False))
        else:
            self.head = nn.Linear(mid_feature_dims, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        # self.fix_init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            # rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def encoder(self, x):
        self.B = x.shape[0]
        # if self.use_moe=='channelmoev3':
        #     embedd_inputs = [patch_embed(x[:, i*self.num_hf:(i+1)*self.num_hf])  for i, patch_embed in enumerate(self.patch_embeds)]
        #     embedd_inputs.append(self.patch_embeds[-1](x[:, -self.num_sf:]))
            
        # else:
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.encoder_mlp:
            x = self.encoder_mlp_layer(x)
        
        if self.model_type == 'VAMoE':
            x = x.reshape(self.B, self.h, self.w, self.embed_dim)

        return x
    
    def decoder(self, x):
        if self.model_type == 'VAMoE':
            x = rearrange(x, 'b h w c -> b (h w) c')

        if self.decoder_mlp:
            x = self.decoder_mlp_layer(x)

        if self.use_moe=='channelmoev3':
            outputs = []
            for head in self.heads:
                x0 = head(x)
                x0 = rearrange(
                    x0,
                    "b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                    p1=self.patch_size[0],
                    p2=self.patch_size[1],
                    h=self.img_size[0] // self.patch_size[0],
                    w=self.img_size[1] // self.patch_size[1],
                )
                outputs.append(x0)
            
            # if self.surface_features != []:
            #     x0 = self.heads[-1](x)
            #     x0 = rearrange(
            #         x0,
            #         "b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            #         p1=self.patch_size[0],
            #         p2=self.patch_size[1],
            #         h=self.img_size[0] // self.patch_size[0],
            #         w=self.img_size[1] // self.patch_size[1],
            #     )
            #     outputs.append(x0)
            output = torch.concat(outputs, dim=1)
            return output
        
        else:
            x = self.head(x)
            x = rearrange(
                x,
                "b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                h=self.img_size[0] // self.patch_size[0],
                w=self.img_size[1] // self.patch_size[1],
            )
            return x

    def expert_loss(self, expert_outputs, targets):
        with torch.no_grad():
            preds = []
            for output, head in zip(expert_outputs, self.heads):
                # print('expert output:', output.shape, head.weight.shape)
                pred = head(output)
                pred = rearrange(
                        pred,
                        "b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                        p1=self.patch_size[0],
                        p2=self.patch_size[1],
                        h=self.img_size[0] // self.patch_size[0],
                        w=self.img_size[1] // self.patch_size[1],
                    )
                preds.append(pred)
            preds = torch.cat(preds, dim=1)
        
        return torch.nn.functional.mse_loss(preds, targets)



    def forward(self, x, target=None, posembed=None, run_mode='train', expert_weight=0.01):
        self.run_mode = run_mode

        x = self.encoder(x)
        recons = self.decoder(x)
        
        # print('encoder feature output, ', x.shape)

        if self.use_moe == 'densemoe' or self.use_moe=='channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
            posembed = self.posembedder(posembed)

        loss = 0
        for i, blk in enumerate(self.blocks):
            if self.model_type == 'vit':
                # if i == 1:
                # a = (x, self.h, self.w, posembed)
                # mid_output = checkpoint(blk, *a, use_reentrant=True)
                # else:
                mid_output = blk(x, self.h, self.w, posembed=posembed)
            elif self.model_type == 'VAMoE':
                mid_output = blk(x)
                # a = (x)
                # mid_output = checkpoint(blk, *a)

            if self.use_moe=='moe':
                x, l = mid_output
                loss += l
            elif self.use_moe=='channelmoev3':
                x, expert_output = mid_output
                if run_mode=='train':
                    if (i+1)%self.num_interval == 0:
                        j = int( (i+1)//self.num_interval ) - 1
                        loss += self.tiny_loss_weights[j] * self.expert_loss(expert_output, target)
                    # print('loss:', loss)
            else:
                x = mid_output.clone()

            # print('mid output: ', x.shape, x[0,0,:10])

        # print('mid feature output, ', x.shape)
        
        output = self.decoder(x)

        if self.loss_type == 'trainl2':
            total_loss = (1-self.loss_weight) * self.loss_gen(output, target) + self.loss_weight * self.loss_recons(recons, target)
            # print('total loss:', total_loss, 'loss:', loss)
            if self.use_moe=='moe':
                total_loss += loss
            if self.use_moe=='channelmoev3' and self.run_mode=='train':
                total_loss += loss
            return output, recons, total_loss

        if self.use_moe=='moe':
            return output, recons, loss
        elif self.use_moe=='channelmoev3' and self.run_mode=='train':
            return output, recons, loss
        else:
            return output, recons


if __name__ == "__main__":
    params = {
        'model_type': 'vit',
        'h_size': 128,
        'w_size': 256,
        'patch_size': 2,
        'feature_dims': 44,
        'num_blocks': 16,
        'encoder_depths': 24,
        'embed_dim': 768,
        'use_moe': 'channelmoe',
        'num_exports': 4,
        'topk': 2,
        'encoder_mlp': False,
        'decoder_mlp': False,
        'loss': 'trainl2',
        'loss_weight': 0.1,
        'use_cl': False,
        'surface_features': ['t2m', 'u10', 'v10', 'msl', 'sp'], 
        # 'surface_features': [], 
        # 'higher_features': ['z', 'q', 'u', 'v', 't'],
        'higher_features': ['z', 'q', 'u'],
        'pressure_level': [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0],
    }
    model = VAMoE(params).to('cuda:0')

    from fvcore.nn.parameter_count import parameter_count_table
    from fvcore.nn.flop_count import flop_count
    print('end111')
    print(parameter_count_table(model))
    dump_input = torch.rand(
        (1, 44, 128, 256)
    ).to('cuda:0')

    sample = torch.randn(1, 44, 128, 256).to('cuda:0')
    target = torch.randn(1, 44, 128, 256).to('cuda:0')

    posembed = torch.ones(4, 44).to('cuda:0')

    # print(flop_count(model, dump_input, ))

    result = model(sample, target=target, posembed=posembed)
    print(result[0].shape, result[1].shape, result[2])
    # print(result[0].shape, result[1].shape)
