import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import sparse
import scipy
import scipy.sparse as sp
import scipy.special

INF = 1e8
"""
REMEMBER: bias(of dense layer) must be 0 in self-attention
        : mask input while generate and forward
"""


def scaled_dot_product(q, k, v, mask=None):
    dim = q.shape[-1]
    attn = jnp.matmul(q, k.swapaxes(-1, -2)) / dim ** -0.5
    if mask is not None:
        attn = jnp.where(mask == 0, -INF, attn)

    attn = nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, v)
    return out, attn


class LinNorm(nn.Module):
    n_dim: int

    def setup(self):
        self.lin1 = nn.Dense(self.n_dim * 4,
                             kernel_init=nn.initializers.xavier_uniform(),
                             bias_init=nn.initializers.zeros)

        self.act = nn.relu
        self.lin2 = nn.Dense(self.n_dim,
                             kernel_init=nn.initializers.xavier_uniform(),
                             bias_init=nn.initializers.zeros)

        self.layer_norm = nn.LayerNorm()

    def __call__(self, X):
        out = self.lin1(X)
        out = self.act(out)
        out = self.lin2(out) + X
        out = self.layer_norm(out)
        return out


class MultiHeadAttention(nn.Module):
    n_dim: int
    n_head: int
    enc_out: bool

    def setup(self):
        if self.enc_out:
            self.q_proj = nn.Dense(self.n_dim * self.n_head,
                                   kernel_init=nn.initializers.xavier_uniform(),
                                   bias_init=nn.initializers.zeros)

            self.kv_proj = nn.Dense(self.n_dim * self.n_head * 2,
                                    kernel_init=nn.initializers.xavier_uniform(),
                                    bias_init=nn.initializers.zeros)
        else:
            self.qkv_proj = nn.Dense(self.n_dim * self.n_head * 3,
                                     kernel_init=nn.initializers.xavier_uniform(),
                                     bias_init=nn.initializers.zeros)

        self.o_proj = nn.Dense(self.n_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

        self.layer_norm = nn.LayerNorm()

    def __call__(self, X, enc_out=None, mask=None):
        """
        X: [bs, seq_len, n_dim]
        """
        bs, seq_len, n_dim = X.shape
        if self.enc_out:
            q = self.q_proj(X)
            kv = self.kv_proj(enc_out)
            k, v = jnp.array_split(kv, 2, axis=-1)
        else:
            qkv = self.qkv_proj(X)
            q, k, v = jnp.array_split(qkv, 3, axis=-1)  # [bs, seq_len, n_head, n_dim]

        q = q.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)  # [bs, n_head, seq_len, n_dim]
        k = k.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)
        v = v.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)

        out, attn = scaled_dot_product(q, k, v, mask)
        out = out.swapaxes(1, 2).reshape(bs, seq_len, self.n_head * n_dim)
        out = X + self.o_proj(out)
        out = self.layer_norm(out)
        return out


class EncoderLayer(nn.Module):
    conf: dict

    def setup(self):
        self.attn = MultiHeadAttention(self.conf["n_dim"], self.conf["n_head"], False)
        self.lin_norm = LinNorm(self.conf["n_dim"])

    def __call__(self, X):
        out = self.attn(X)
        out = self.lin_norm(out)
        return out


class DecoderLayer(nn.Module):
    conf: dict

    def setup(self):
        self.mask_attn = MultiHeadAttention(self.conf["n_dim"], self.conf["n_head"], False)
        self.enc_dec_attn = MultiHeadAttention(self.conf["n_dim"], self.conf["n_head"], True)
        self.lin_norm = LinNorm(self.conf["n_dim"])

    def __call__(self, X, enc_out, mask):
        out = self.mask_attn(X, mask=mask)
        out = self.enc_dec_attn(X, enc_out=enc_out)
        out = self.lin_norm(out)
        return out


class PredLayer(nn.Module):
    conf: dict

    def setup(self):
        self.n_token = self.conf["n_token"]
        self.beta = self.conf["beta"]
        self.lin = nn.Dense(self.n_token,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)

    def __call__(self, X):
        out = self.lin(X)
        bs, seq_len, n_token = out.shape
        noise = np.random.rand(bs, seq_len, n_token) * self.beta
        logits = out + noise
        return logits


class Transformer(nn.Module):
    conf: dict
    # ui_graph: sp.coo_matrix

    def setup(self):
        self.item_dst = nn.Embed(self.conf["n_token"], self.conf["n_dim"])
        self.item_src = nn.Embed(self.conf["n_token"], self.conf["n_dim"])
        # self.item_emb = nn.Embed(self.conf["n_token"], self.conf["n_dim"])
        self.encoder = [EncoderLayer(self.conf) for _ in range(self.conf["n_layer"])]
        self.decoder = [DecoderLayer(self.conf) for _ in range(self.conf["n_layer"])]
        self.pred_layer = PredLayer(self.conf)
        self.get_mask()

    def get_mask(self, diag=0):
        m = self.conf["seq_len"]
        self.dec_mask = jnp.tril(jnp.ones((m, m)), k=diag)
    
    def __call__(self, X, test=False):
        if not test:
            logits = self.forward(X)
            return logits
        else:
            # bundle_out = self.beam_search_gen(X, beam_width=3)
            bundle_out = self.bs_generate(X)
            return bundle_out

    def forward(self, X):
        enc_ids, dec_ids = X
        bs = enc_ids.shape[0]
        enc_pad_mask = 1 - jnp.array(enc_ids == self.conf["pad"], dtype=jnp.float32) \
            .reshape(bs, self.conf["seq_len"], 1)

        dec_pad_mask = 1 - jnp.array(dec_ids == self.conf["pad"], dtype=jnp.float32) \
            .reshape(bs, self.conf["seq_len"], 1)

        enc_inp = self.item_src(enc_ids) * enc_pad_mask
        dec_inp = self.item_dst(dec_ids) * dec_pad_mask
        # enc_inp = self.item_emb(enc_ids) * enc_pad_mask
        # dec_inp = self.item_emb(dec_ids) * dec_pad_mask

        enc_out = enc_inp
        for l in self.encoder:
            enc_out = l(enc_out)

        # print("FORWARD_INP:", enc_inp)
        # print("FORWARD_ENC_OUT:", enc_out)

        dec_out = dec_inp
        for l in self.decoder:
            dec_out = l(dec_out, enc_out, self.dec_mask)

        logits = self.pred_layer(dec_out)
        return logits

    def generate(self, X):
        """
        greedy take best from best ...
        while test bs = 1
        """
        bs = X.shape[0]  # 1
        enc_ids, dec_ids = X, [self.conf["sos"]]
        enc_pad_mask = 1 - jnp.array(enc_ids == self.conf["pad"], dtype=jnp.float32) \
            .reshape(bs, self.conf["seq_len"], 1)
        enc_inp = self.item_src(enc_ids) * enc_pad_mask
        # enc_inp = self.item_emb(enc_ids) * enc_pad_mask

        enc_out = enc_inp
        for l in self.encoder:
            enc_out = l(enc_out)

        last = 1
        timestep = 0

        while last != self.conf["eos"] and last != self.conf["pad"] \
                and len(dec_ids) < self.conf["seq_len"]:

            padding = [self.conf["pad"]] * (self.conf["seq_len"] - len(dec_ids))
            dec_ids_temp = jnp.array(dec_ids + padding)
            # dec_inp = jnp.array([self.item_emb(dec_ids_temp)])  # wrap ids to [bs, seq_len]
            dec_inp = jnp.array([self.item_dst(dec_ids_temp)])  # wrap ids to [bs, seq_len]
            dec_pad_mask = 1 - jnp.array(dec_ids_temp == self.conf["pad"], dtype=jnp.float32) \
                .reshape(bs, self.conf["seq_len"], 1)
            dec_out = dec_inp * dec_pad_mask
            for l in self.decoder:
                dec_out = l(dec_out, enc_out, self.dec_mask)

            logits = self.pred_layer(dec_out).squeeze()
            _, pred_id = jax.lax.top_k(logits, k=1)
            last = pred_id[timestep][0]
            if int(last) != self.conf["pad"] and int(last) != self.conf["eos"]:
                dec_ids.append(int(last))
            timestep += 1

        if dec_ids[-1] == self.conf["eos"]:
            dec_ids = dec_ids[:-1]  # !!! important data id + 3 sos, eos, pad
        gen_bun = [np.array(dec_ids[1:]) - 3] # list of list bundle [-3 token pad start end]
        return gen_bun
    

    def beam_search_gen(self, X, beam_width=3):
        """
        take K best result from K*K results
        while test bs = 1
        """
        bs = X.shape[0]  # 1
        enc_ids, buns = X, [[self.conf["sos"]] for i in range(beam_width)]
        enc_out = self.get_enc_out(enc_ids)

        temp_storage = [] # size K * K when timestep > 1
        top_k_prob = [1. for i in range(beam_width)]

        partial_bun = [1]
        timestep = 0
        prob, pred_id = self.get_topk_out(enc_out, partial_bun, timestep, beam_width)
        # timestep == 0
        for i in range(beam_width):
            temp_storage.append(buns[i] + [pred_id[i].tolist()])
            top_k_prob[i]*=prob[i]
        timestep+=1

        # timestep > 0
        check_end = 1
        while check_end:
            new_temp_storage = []
            new_topk_prob = []
            continue_check = 0
            for bid in range(0, len(temp_storage)):
                local_continue_check = 0
                pbun = temp_storage[bid]
                pprob = top_k_prob[bid]
                new_buns = []
                new_probs = []
                prob, iidt = self.get_topk_out(enc_out, pbun, timestep=timestep, beam_width=beam_width)
                # construct k potential next index of pbun
                for i in range(0, beam_width):
                    next_iid = int(iidt[i])
                    if next_iid == self.conf["pad"] or next_iid == self.conf["eos"] \
                        or len(pbun) >= self.conf['seq_len']:
                        new_buns.append(pbun.copy())
                        new_probs.append(prob[i])
                        ## stop gen next iid
                    else:
                        new_bun = pbun.copy()
                        new_bun.append(next_iid)
                        new_buns.append(new_bun)
                        new_probs.append(prob[i] * pprob)
                        local_continue_check = 1

                new_temp_storage.extend(new_buns)
                new_topk_prob.extend(new_probs)

                continue_check+=local_continue_check

            if continue_check == 0:
                check_end=0

            temp_storage = []
            new_topk_prob = jnp.array(new_topk_prob)
            nprob, nvalidid = jax.lax.top_k(new_topk_prob, k=beam_width)
            for valid_bid in nvalidid:
                temp_storage.append(new_temp_storage[valid_bid])

            timestep+=1

        gen_buns = []
        for i in temp_storage:
            i.sort()
            gen_buns.append(i[1:] - 3)
        return gen_buns
    

    def bs_generate(self, X):
        # X: [bs, seq_len] to encoder
        # get encoedr out
        bs = X.shape[0]
        enc_ids, dec_ids = X, jnp.full(shape=(bs, 1), fill_value=self.conf["sos"])
        enc_out = self.bs_get_enc_out(enc_ids)
        #get decoder out
        for pos in range(0, self.conf["seq_len"]):
            padding = jnp.full(shape=(bs, self.conf["seq_len"] - dec_ids.shape[1]), fill_value=self.conf["pad"])
            dec_inp_ids = np.concatenate([dec_ids, padding], axis=1)
            # dec_inp = self.item_emb(dec_inp_ids) #[bs, seq_len, n_dim]
            dec_inp = self.item_dst(dec_inp_ids)

            dec_pad_mask = 1 - jnp.array(dec_inp_ids == self.conf["pad"], dtype=jnp.float32) \
                .reshape(bs, self.conf["seq_len"], 1) #pad_mask con broadcast to [bs, seq_len, n_dim]
            
            dec_out = dec_inp * dec_pad_mask

            for l in self.decoder:
                dec_out = l(dec_out, enc_out, self.dec_mask)

            logits = self.pred_layer(dec_out).squeeze()
            logits_time = logits[:, pos]
            # print(logits.shape)
            # print(logits_time.shape)
            _, pred_id = jax.lax.top_k(logits_time, k=1)
            # pred_id = pred_id
            dec_ids = np.concatenate([dec_ids, pred_id], axis=1) # while propagating not -3
            # print(dec_ids)
        dec_ids = dec_ids[:, 1:] # rm sos token
        # print(dec_ids)
        dec_ids = dec_ids - 3
        print(dec_ids)
        return dec_ids
        # exit()


    def bs_get_enc_out(self, enc_ids):
        # print(enc_ids.shape)
        bs = enc_ids.shape[0]
        enc_pad_mask = 1 - jnp.array(enc_ids == self.conf["pad"], dtype=jnp.float32) \
            .reshape(bs, self.conf["seq_len"], 1) # [bs, seq_len, 1] -> can broadcast to [bs, seq_len, n_dim]
        enc_inp = self.item_src(enc_ids) * enc_pad_mask #[bs, seq_len, n_dim]
        # enc_inp = self.item_emb(enc_ids) * enc_pad_mask

        enc_out = enc_inp
        for l in self.encoder:
            enc_out = l(enc_out)
        return enc_out
        

    def get_enc_out(self, enc_ids):
        bs = 1
        enc_pad_mask = 1 - jnp.array(enc_ids == self.conf["pad"], dtype=jnp.float32) \
            .reshape(bs, self.conf["seq_len"], 1)
        enc_inp = self.item_src(enc_ids) * enc_pad_mask
        # enc_inp = self.item_emb(enc_ids) * enc_pad_mask

        enc_out = enc_inp
        for l in self.encoder:
            enc_out = l(enc_out)
        return enc_out
    

    def get_topk_out(self, enc_out, dec_ids, timestep, beam_width):
        bs = 1
        dec_pad = [self.conf["pad"]] * (self.conf["seq_len"] - len(dec_ids))
        dec_inp_id = jnp.array(dec_ids + dec_pad)
        dec_inp = jnp.array([self.item_dst(dec_inp_id)])
        # dec_inp = jnp.array([self.item_emb(dec_inp_id)])
        dec_pad_mask = 1 - jnp.array(dec_inp_id == self.conf["pad"], dtype=jnp.float32) \
                    .reshape(bs, self.conf["seq_len"], 1)
                
        dec_out = dec_inp * dec_pad_mask
        for l in self.decoder:
            dec_out = l(dec_out, enc_out, self.dec_mask)

        pred = self.pred_layer(dec_out).squeeze() # squeeze batch size = 1 :)
        pred = nn.softmax(pred)
        prob, topk_ids = jax.lax.top_k(pred, k=beam_width)
        prob = prob[timestep]
        topk_ids = topk_ids[timestep]
        return prob, topk_ids


class BRIDGE(nn.Module):
    conf: dict
    ui_graph: sp.coo_matrix

    def setup(self):
        self.generator = Transformer(self.conf)
        self.item_ranking_emb = nn.Embed(self.conf["n_item"], self.conf["n_dim"])
        self.all_item_idx = jnp.arange(0, self.conf["n_item"])
        self.iui_graph = self.create_graph()

    def create_graph(self):
        """
        LightGCN on IUI propagation graph
        """
        ui_graph = self.ui_graph.tocoo()
        iui_graph = ui_graph.T @ ui_graph
        iui_graph = (iui_graph - iui_graph * sp.eye(self.conf["n_item"]).tocoo()) > 0 # [0, 1] type
        norm1 = sp.diags(1 / (np.sqrt(iui_graph.sum(axis=1).A.ravel()) + 1e-8))
        norm0 = sp.diags(1 / (np.sqrt(iui_graph.sum(axis=0).A.ravel()) + 1e-8))
        iui_graph = sparse.BCOO.from_scipy_sparse(norm1 @ iui_graph @ norm0)
        return iui_graph
    
    def __call__(self, X, test=False):
        if not test:
            logits = self.generator(X, test)
            item_out = self.item_ranking_emb(self.all_item_idx)
            item_out = (item_out + self.iui_graph @ item_out) / 2
            # print(item_out.shape)
            denom = jnp.linalg.norm(item_out, axis=1).reshape(-1, 1)
            item_out /= denom
            return logits, item_out
        else:
            bundle_out = self.generator(X, test)
            return bundle_out
        
