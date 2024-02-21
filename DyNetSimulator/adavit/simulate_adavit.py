import sys
sys.path.append('.')
sys.path.append('..')

from hardware_models.predictor_transformer import PredictorTransformer

def simulate_token_transformer(B, L, head_num, in_dim, out_dim, mlp_ratio):
    # layer norm
    # qkv projection
    # q @ k
    # softmax
    # @ v
    # projection
    # add
    # layer norm
    # mlp
    
    r_layer_norm_in = predictor.simulate_layernorm(shape=(B, L, in_dim))
    
    # attention block
    r_qkv = predictor.simulate_linear(x_shape=(B, L, in_dim), w_shape=(3*out_dim, in_dim), out_shape=(B, L,3*out_dim))
    r_att_matrix = predictor.simulate_matmul(a_shape=(B, head_num, L, out_dim//head_num), b_shape=(B, head_num, L, out_dim//head_num), out_shape=(B, head_num, L,L))
    r_softmax = predictor.simulate_softmax(shape=(B, head_num, L,L))
    r_att_mult_v = predictor.simulate_matmul(a_shape=(B, head_num, L,L), b_shape=(B, head_num, L, in_dim//head_num), out_shape=(B, head_num, L,in_dim))
    r_proj = predictor.simulate_linear(x_shape=(B, L, in_dim), w_shape=(out_dim, in_dim), out_shape=(B, L,out_dim))
    r_add = predictor.simulate_elementwise(shape=(B, L,out_dim))
    
    r_norm2 = predictor.simulate_layernorm(shape=(B, L,out_dim))
    
    # MLP
    r_mlp_fc1 = predictor.simulate_linear(x_shape=(B, L, out_dim), w_shape=(out_dim*mlp_ratio, out_dim), out_shape=(B, L,out_dim*mlp_ratio))
    r_mlp_gelu = predictor.simulate_gelu(shape=(B, L,out_dim*mlp_ratio))
    r_mlp_fc2 = predictor.simulate_linear(x_shape=(B, L,out_dim*mlp_ratio), w_shape=(out_dim, out_dim*mlp_ratio), out_shape=(B, L,out_dim))
    
    return r_layer_norm_in + r_qkv + r_att_matrix + r_softmax + r_att_mult_v + r_proj + r_add + r_norm2 + r_mlp_fc1 + r_mlp_gelu + r_mlp_fc2
    
def simulate_token_performer(B, L, in_dim, out_dim, kernel_ratio=0.5):
    r_norm1 = predictor.simulate_layernorm(shape=(B, L, in_dim)).latency
    r_qkv = predictor.simulate_linear(x_shape=(B, L, in_dim), w_shape=(3*out_dim, in_dim), out_shape=(B, L,3*out_dim)).latency
    m = int(out_dim * kernel_ratio)
    r_kp_qp = 2 * (
        predictor.simulate_elementwise(shape=(B,L,out_dim)).latency + 
        predictor.simulate_linear(x_shape=(B, L, out_dim), w_shape=(m, out_dim), out_shape=(B, L, m)).latency + 
        predictor.simulate_add(B, L, m).latency + 
        predictor.simulate_elementwise(shape=(B,L,m)).latency
    )
    
    r_D = predictor.simulate_linear(x_shape=(B, L, m), w_shape=(1, m), out_shape=(B, L, 1)).latency
    r_kptv = predictor.simulate_matmul(a_shape=(B, out_dim, L), b_shape=(B, L, m), out_shape=(B, out_dim, m)).latency
    r_y = predictor.simulate_matmul(a_shape=(B, L, m), b_shape=(B, m, out_dim), out_shape=(B, L, out_dim)).latency
    r_proj = predictor.simulate_linear(x_shape=(B, L, out_dim), w_shape=(out_dim, out_dim), out_shape=(B, L, out_dim)).latency
    
    r_norm2 = predictor.simulate_layernorm(shape=(B, L, out_dim)).latency
    r_mlp = predictor.simulate_linear(x_shape=(B, L, out_dim), w_shape=(out_dim, out_dim), out_shape=(B, L, out_dim)).latency + \
        predictor.simulate_gelu(shape=(B, L, out_dim)).latency + \
            predictor.simulate_linear(x_shape=(B, L, out_dim), w_shape=(out_dim, out_dim), out_shape=(B, L, out_dim)).latency
            
    return r_norm1 + r_qkv + r_kp_qp + r_D + r_kptv + r_y + r_proj + r_norm2 + r_mlp

def simulate_t2t_module(B, dim=64, head_num=7):
    r_split0 = predictor.simulate_unfold(in_shape=(B,3,224,224), out_shape=(B,147,56,56)).latency
    # r_attention1 = simulate_token_transformer(B=B, L=56*56, in_dim=147, out_dim=dim, mlp_ratio=1)
    r_attention1 = simulate_token_performer(B=B, L=56*56, in_dim=147, out_dim=dim, kernel_ratio=0.5)
    
    r_split1 = predictor.simulate_unfold(in_shape=(B,dim,56,56), out_shape=(B,dim*9,28,28)).latency
    # r_attention2 = simulate_token_transformer(B=B, L=28*28, in_dim=dim*9, out_dim=dim, mlp_ratio=1)
    r_attention2 = simulate_token_performer(B=B, L=28*28, in_dim=dim*9, out_dim=dim, kernel_ratio=0.5)
    
    r_split2 = predictor.simulate_unfold(in_shape=(B,dim,28,28), out_shape=(B,dim*9,14,14)).latency
    r_proj = predictor.simulate_linear(x_shape=(B, 14*14, dim*9), w_shape=(dim*head_num, dim*9), out_shape=(B,196,dim*head_num)).latency
    
    return r_split0 + r_attention1 + r_split1 + r_attention2 + r_split2 + r_proj

def simulate_add_pos_embed(B, L, dim=448):
    return predictor.simulate_elementwise(shape=(B,L,dim)).latency

def simulate_ada_attention(B, L=197, in_dim=448, head_num=7, token_skip=True, token_density=1., head_skip=True, head_density=1.):
    dim_per_head = in_dim // head_num
    
    if head_skip:
        sparse_head_num = int(head_num * head_density)        
        r_qkv = 3 * predictor.simualte_dylinear(x_shape=(B, L, in_dim),
                                                w_shape=(in_dim, in_dim), 
                                                out_shape=(B, L, in_dim), 
                                                ic_density=1,
                                                oc_density=head_density).latency
    else:
        assert head_density == 1.
        sparse_head_num = head_num
        
        r_qkv = 3 * predictor.simulate_linear(x_shape=(B,L,in_dim), w_shape=(in_dim, in_dim), out_shape=(B, L, in_dim)).latency
        # print(r_qkv)
        # assert(0==1)
            
    
    if token_skip:
        r_token_mask = predictor.simualte_dylinear(x_shape=(B, L-1, in_dim),
                                                   w_shape=(1, in_dim), 
                                                   out_shape=(B, L, 1), 
                                                   ic_density=head_density,
                                                   oc_density=1).latency
    else:
        r_token_mask = 0.
        assert token_density == 1.
    
    L_select = int(L * token_density)
    
    r_attn = predictor.simulate_matmul(a_shape=(B, sparse_head_num, L_select, dim_per_head), 
                                       b_shape=(B, sparse_head_num, dim_per_head, L_select ), 
                                       out_shape=(B, sparse_head_num, L_select, L_select)).latency \
             + predictor.simulate_softmax(shape=(B, sparse_head_num, L_select, L_select)).latency
    
    r_v = predictor.simulate_matmul(a_shape=(B, sparse_head_num, L_select, L_select),
                                    b_shape=(B, sparse_head_num, L_select, dim_per_head),
                                    out_shape=(B, sparse_head_num, L_select, dim_per_head)).latency
    
    if head_skip:
        r_proj = predictor.simualte_dylinear(x_shape=(B, L_select, in_dim),
                                             w_shape=(in_dim, in_dim), 
                                             out_shape=(B, L_select, in_dim), 
                                             ic_density=head_density,
                                             oc_density=head_density).latency
    else:
        r_proj = predictor.simulate_linear(x_shape=(B, L_select, in_dim), 
                                           w_shape=(in_dim, in_dim), 
                                           out_shape=(B, L_select, in_dim)).latency
    
    # print(f'r_qkv:{r_qkv}, r_token_mask:{r_token_mask}, r_attn:{r_attn}, r_v:{r_v}, r_proj:{r_proj}')
    
    return r_qkv + r_token_mask + r_attn + r_v + r_proj, L_select
    
def simulate_ada_mlp(B, L, in_dim, mlp_ratio, head_skip, head_num, head_density):
    hidden_dim = in_dim * mlp_ratio
    if not head_skip:
        assert head_density == 1.
        r_fc1 = predictor.simulate_linear(x_shape=(B, L, in_dim), w_shape=(hidden_dim, in_dim), out_shape=(B, L, hidden_dim)).latency
    else:
        # dim_per_head = in_dim // head_num
        # sparse_in_dim = dim_per_head * int(head_num * head_density)
        r_fc1 = predictor.simualte_dylinear(x_shape=(B, L, in_dim), 
                                            w_shape=(hidden_dim, in_dim),
                                            out_shape=(B, L, hidden_dim),
                                            ic_density=head_density,
                                            oc_density=1.).latency
    
    r_act = predictor.simulate_gelu(shape=(B, L, hidden_dim)).latency
    r_fc2 = predictor.simulate_linear(x_shape=(B, L, hidden_dim), w_shape=(in_dim, hidden_dim), out_shape=(B, L, in_dim)).latency
    return r_fc1 + r_act + r_fc2
    
def simulate_ada_block(B=1, L=197, in_dim=448, mlp_ratio=3, token_skip=True, token_density=1., head_skip=True, head_num=7, head_density=1., layer_skip=True, layer_density_attn=1., layer_density_mlp = 1.):
    if layer_skip:
        r_layer_policy = predictor.simulate_linear(x_shape=(B,in_dim), w_shape=(2, in_dim), out_shape=(B,2)).latency
    else:
        r_layer_policy = 0
        assert (layer_density_attn == 1. and layer_density_mlp == 1.)
        
    if head_skip:
        r_head_policy = predictor.simulate_linear(x_shape=(B,in_dim), w_shape=(head_num, in_dim), out_shape=(B,head_num)).latency
    else:
        r_head_policy = 0
        assert head_density == 1.
    
    r_attn, L_select = simulate_ada_attention(B, L, in_dim, head_num, token_skip, token_density, head_skip, head_density=head_density)
    
    sparse_dim = int(in_dim*head_density)
    
    # print(r_attn)
    # assert(0==1)
    
    r_attn_block = layer_density_attn * (
        predictor.simulate_layernorm(shape=(B, L, in_dim)).latency + 
        r_attn + 
        predictor.simulate_add(B, L_select, sparse_dim).latency
    )
    
    r_mlp_block = layer_density_mlp * (
        predictor.simulate_layernorm(shape=(B, L, in_dim)).latency + 
        simulate_ada_mlp(B, L_select, in_dim, mlp_ratio, head_skip, head_num, head_density) + 
        predictor.simulate_add(B, L_select, in_dim).latency
    )
    
    return r_layer_policy + r_head_policy + r_attn_block + r_mlp_block


def simulate_tail(B, dim=448):
    r_layernorm = predictor.simulate_layernorm(shape=(B,197,dim)).latency
    r_fc = predictor.simulate_linear(x_shape=(B,dim), w_shape=(1000, dim), out_shape=(B,1000)).latency
    
    return r_layernorm + r_fc

