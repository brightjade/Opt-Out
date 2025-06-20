from copy import deepcopy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        local_files_only=args.local_files_only,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(args, ckpt_path=None):
    if ckpt_path:
        print(f"Loading model from {ckpt_path}...")
    else:
        print(f"Loading model from {args.model_name_or_path}...")

    model = AutoModelForCausalLM.from_pretrained(
                ckpt_path if ckpt_path else args.model_name_or_path,
                torch_dtype=args.dtype if not args.do_train else None,
                device_map="auto",
                use_cache=False if args.do_train else True,
                attn_implementation=args.attn_implementation,
                cache_dir=args.cache_dir if args.cache_dir else None,
                local_files_only=args.local_files_only,
            )
    return model


def create_reference_model(main_model):
    model = deepcopy(main_model).bfloat16()
    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False
    return model.eval()


def sliced_wasserstein_distance(W_i, W_c, n_projections=100, p=2):
    r""" Computes a Monte-Carlo approximation of the p-Sliced Wasserstein distance between two weights

    Parameters
    ----------
    W_i: array-like, shape (n, d)
        Weights of the initial parameters (frozen)
    W_c: array-like, shape (m, d)
        Weights of the current parameters
    n_projections: int, optional (default=100)
        Number of random projections
    p: int, optional (default=2)
        Power of the Wasserstein distance

    Returns
    -------
    swd: float
        Sliced Wasserstein Distance between the two weights
    """
    # Generate random projections
    projections = torch.randn(W_i.shape[1], n_projections)
    projections = projections / torch.sqrt((projections**2).sum(dim=0))

    # Project weights to lower dimension
    W_i_projections = torch.mm(W_i, projections.to(W_i))
    W_c_projections = torch.mm(W_c, projections.to(W_c))

    # Sort the weights
    u_values, _ = torch.sort(W_i_projections, dim=0)
    v_values, _ = torch.sort(W_c_projections, dim=0)

    # Compute Wasserstein distances
    wasserstein_distances = torch.abs(u_values - v_values.to(W_i))
    wasserstein_distances = (wasserstein_distances ** p).sum(dim=0)

    # Compute p-sliced SWD
    swd = wasserstein_distances.mean()

    return swd
