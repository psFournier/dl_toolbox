import torch.nn as nn
import timm


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
        
class DecoderLinear(nn.Module):
    def __init__(self, num_classes, patch_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.head = nn.Linear(self.embed_dim, num_classes)
        # self.apply applies to every submodule recursively, typical for init
        self.apply(init_weights)

    def no_weight_decay(self):
        return set()

    def forward(self, x, img_height):
        x = self.head(x) # BxNpatchxEmbedDim -> BxNpatchxNcls
        num_patch_h = img_height//self.patch_size # num_patch_per_axis Np
        # rearrange auto computes h & w from the fixed val h; C-order enum used, see docs
        x = rearrange(x, "b (h w) c -> b c h w", h=num_patch_h) # BxNclsxNpxNp
        return x

class SegmenterLinear(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        weights,
        norm_layer=None,
        **kwargs
    ):  
        
        self.backbone = timm.create_model(
            backbone, # Name of the pretrained model
            pretrained=True,
            dynamic_img_size=True # Deals with image sizes != from pretraining
        )
        self.decoder = DecoderLinear(
            num_classes,
            self.backbone.patch_embed.patch_size[0], # patch_size
            self.backbone.embed_dim
        )
        
        super(ViT,self).__init__(**kwargs)
        assert in_channels==3
        self.weights = weights
        self.num_classes = num_classes
        self.model = ViTForImageClassification.from_pretrained(weights, num_labels=num_classes)
        self.feature_extractor = self.model.vit

    def forward(self, x):
        H, W = im.size(2), im.size(3)
        x = self.backbone.forward_features(im)
        # We ignore non-patch tokens when recovering a segmentation by decoding
        x = x[:,self.backbone.num_prefix_tokens:,...]
        masks = self.decoder(x, H)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        return masks
    
