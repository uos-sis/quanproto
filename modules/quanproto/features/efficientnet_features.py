import timm
import torch
from torchsummary import summary
from transformers import AutoModel

model_urls = {
    "efficientnet-b0": "google/efficientnet-b0",
}


# Check if the weights are the same
def compare_weights(model1, model2):
    # Ensure both models are in the same mode (eval or train)
    model1.eval()
    model2.eval()

    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(param1, param2):
            print("Weights are different!")
            return False
    print("Weights are the same!")
    return True


def efficientnet_b0_features(pretrained=False, with_pooling=False):
    if pretrained:
        return AutoModel.from_pretrained(
            model_urls["efficientnet-b0"], cache_dir="./model_cache"
        )

        model = timm.create_model("efficientnet_b0", pretrained=True)
        # remove (classifier) and (global_pool) layers
        if not with_pooling:
            model.global_pool = torch.nn.Identity()
        model.classifier = torch.nn.Identity()
    else:
        return AutoModel.from_pretrained(
            model_urls["efficientnet-b0"], cache_dir="./model_cache"
        )

        model = timm.create_model("efficientnet_b0", pretrained=False)
        # remove (classifier) and (global_pool) layers
        if not with_pooling:
            model.global_pool = torch.nn.Identity()
        model.classifier = torch.nn.Identity()

    return model


if __name__ == "__main__":
    model = efficientnet_b0_features(pretrained=True, with_pooling=False)
    summary(model, (3, 224, 224), device="cpu")
    model2 = AutoModel.from_pretrained(model_urls["efficientnet-b0"])

    compare_weights(model, model2)
