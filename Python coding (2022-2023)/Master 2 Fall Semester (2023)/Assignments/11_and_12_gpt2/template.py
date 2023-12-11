import numpy as np
from tqdm import tqdm

from encoder import get_encoder
from tools import get_params


# Multi-head attention

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    sum_exp = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / sum_exp


def attention(Q, K, V):
    inside = (Q @ K.T) / np.sqrt(Q.shape[1])
    a = softmax(inside)

    return (a @ V)


def masked_attention(Q, K, V, mask):
    inside_a = (Q @ K.T) / np.sqrt(Q.shape[1])
    masked_product = inside_a + mask
    a = softmax(masked_product)
    
    return (a @ V)


def linear_projection(x, w, b):
    return (x @ w) + b


def multi_head_attention(x, attn, number_of_heads):
    w_1, b_1 = attn["c_attn"]["w"], attn["c_attn"]["b"]
    w_2, b_2 = attn["c_proj"]["w"], attn["c_proj"]["b"]
    mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    """
        Your code here
    """
    
    # First linear projection
    proj_q = linear_projection(x, w_1[:, :w_1.shape[1]//3], b_1)
    proj_k = linear_projection(x, w_1[:,w_1.shape[1]//3:2*w_1.shape[1]//3], b_1)
    proj_v = linear_projection(x, w_1[:,2*w_1.shape[1]//3:], b_1)
    
    # Split each matrix into heads
    proj_q = np.split(proj_q, number_of_heads, axis=1)
    proj_k = np.split(proj_k, number_of_heads, axis=1)
    proj_v = np.split(proj_v, number_of_heads, axis=1)
    
    # Perform masked attention over each head
    outputs = []
    for i in range(number_of_heads):
        out = masked_attention(proj_q[i], proj_k[i], proj_v[i], mask)
        outputs.append(out)
    
    # Merge heads horizontally
    merged = np.concatenate(outputs, axis=1)
    
    # Second linear projection
    x = linear_projection(merged, w_2, b_2)
    
    
    return x




# Transformer blocks and GPT2


def gelu(x):
    cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    return x * cdf


def layer_normalization(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    x_norm_scaled = g * x_norm + b
    return x_norm_scaled


def feed_forward_network(x, mlp):
    w_1, b_1 = mlp["c_fc"]["w"], mlp["c_fc"]["b"]
    w_2, b_2 = mlp["c_proj"]["w"], mlp["c_proj"]["b"]
    """
        Your code here
    """
    projection = linear_projection(x, w_1, b_1)
    activation = gelu(projection)
    x = linear_projection(activation, w_2, b_2)
    return x


def transformer_block(x, block, number_of_heads):
    mlp, attn = block["mlp"], block["attn"]
    ln_1, ln_2 = block["ln_1"], block["ln_2"]
    g_1, b_1, g_2, b_2 = ln_1["g"], ln_1["b"], ln_2["g"], ln_2["b"]
    """
        Your code here
    """    
    first_layer = layer_normalization(x, g_1, b_1)
    first_foward_pass = multi_head_attention(first_layer, attn, number_of_heads) + x
    
    second_layer = layer_normalization(first_foward_pass, g_2, b_2)
    x = multi_head_attention(second_layer, attn, number_of_heads) + x

    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, number_of_heads):
    g_final, b_final = ln_f["g"], ln_f["b"]
    x = wte[inputs] + wpe[range(len(inputs))]
    """
        Your code here
    """
    for block in blocks:
        x = transformer_block(x, block, number_of_heads)
    
    x = layer_normalization(x, g_final, b_final)
    
    return x @ wte.T


def generate(input_text, tokens_to_generate=40, model_size="124M", models_dir="models", loading_bar=True):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    
    hparams, params = get_params(model_size, models_dir)
    encoder = get_encoder(model_size, models_dir)
    number_of_heads = hparams["n_head"]
    max_context = hparams["n_ctx"]

    # Port the input text to ids
    input_ids = encoder.encode(input_text)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + tokens_to_generate < max_context

    # generate output ids
    output_ids = []
    
    if loading_bar:
        loop_range = tqdm(range(tokens_to_generate), "Thinking...")
    else:
        loop_range = range(tokens_to_generate)

    for _ in loop_range:
        # Call our gtp2 model with input plus generated tokens
        output = gpt2(input_ids + output_ids, **params, number_of_heads=number_of_heads) 

        # Get the next token from the output
        next_id = np.argmax(output[-1])

        # Save the result
        output_ids.append(int(next_id))

    # Port the output ids to text
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    
    Test your implemetntation with something like this:
    print(generate("Hello! How do you do?"))

    You can try out different sized models from this list: ["124M", "355M", "774M", "1558M"]
    Make sure you have enough space on your device since the bigger models are quite large.
    """
    print("Section 1.1")
    print(softmax(np.array([[-1., 0.], [0.2, 1.]])))
    
    print("\nSection 1.2")
    np.random.seed(4321)
    q = np.random.rand(3,2)
    k = np.random.rand(3,2)
    v = np.random.rand(3,2)
    x = attention(q, k, v)
    print(x)
    
    print("\nSection 1.3")
    np.random.seed(4321)
    nf = 10
    q = np.random.rand(nf,2)
    k = np.random.rand(nf,2)
    v = np.random.rand(nf,2)
    mask = (1 - np.tri(nf)) * -1e10
    x = masked_attention(q, k, v, mask)
    print(x)
    
    print("\nSection 2.1")
    np.random.seed(4321)
    x = np.random.rand(3,2)
    w = np.random.rand(2,3)
    b = np.random.rand(3,1)
    lp = linear_projection(x, w, b)
    print(lp)
    
    print("\nSection 2.2")
    np.random.seed(4321)
    x = np.random.rand(3,4)
    w_1 = np.random.rand(4,12)
    b_1 = np.random.rand(3,1)
    w_2 = np.random.rand(4,3)
    b_2 = np.random.rand(3,1)
    attn = {"c_attn": {"w": w_1, "b": b_1}, "c_proj": {"w": w_2, "b": b_2}}
    x = multi_head_attention(x, attn, 2)
    print(x)

    ##########################################
    print("\nSection 1.1")
    print(gelu(np.array([[-1., 0.], [0.2,  1.]])))
    
    print("\nSection 1.2")
    np.random.seed(4321)
    x = np.random.rand(3,2)
    g = np.random.rand(3,2)
    b = np.random.rand(3,1)
    ln = layer_normalization(x, g, b)
    print(ln)
    
    print("\nSection 2.1")
    np.random.seed(4321)
    x = np.random.rand(3,4)
    w_1 = np.random.rand(4,5)
    b_1 = np.random.rand(3,1)
    w_2 = np.random.rand(5,4)
    b_2 = np.random.rand(3,1)
    mlp = {"c_fc": {"w": w_1, "b": b_1}, "c_proj": {"w": w_2, "b": b_2}}
    x = feed_forward_network(x, mlp)
    print(x)
    
    print("\nSection 2.4")
    g1 = generate("Hello! How are you?")
    print(g1)
    g2 = generate("What is the weather like tomorrow?")
    print(g2)
    g3 = generate("Tell me a story")
    print(g3)