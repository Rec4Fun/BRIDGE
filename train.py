import pickle

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from model import BRIDGE
from flax.training import train_state

from tqdm import tqdm

from config import conf
from utils import *
from argparse import ArgumentParser

from flax.training import orbax_utils

"""
REMEMBER FIX OUTPUT LAYER: n_item to n_token due to need to stop
n_token = 9 -> output 9 units (3 token an 6 item)
to debug by print value pls comment @jax.jit
all item ids += 3
"""


def get_args():
    argp = ArgumentParser()
    argp.add_argument("--device_id", type=int, default=0)
    argp.add_argument("--dataset", type=str,
                      help="which dataset to train", default="clothing")
    argp.add_argument("--beam_width", type=int, default=3)
    argp.add_argument("--batch_size", type=int, default=16)
    argp.add_argument("--seq_len", type=int, default=10)
    argp.add_argument("--n_dim", type=int, default=16)
    argp.add_argument("--n_head", type=int, default=2)
    argp.add_argument("--n_layer", type=int, default=2)
    argp.add_argument("--data_path", type=str, default="data")
    argp.add_argument("--epochs", type=int, default=100)
    # argp.add_argument("--p_epochs", type=int, default=10)
    argp.add_argument("--continue_training", type=bool, default=False)
    argp.add_argument("--weight_path", type=str, default="")
    args = argp.parse_args()
    return args


# @jax.jit
def train_step(state, batch, target, iidata):
    # remember not to pass state to loss fn
    def loss_fn(params, batch, target, iidata):
        logits, item_out = state.apply_fn(params, batch)

        iid, pid, nid = iidata
        
        item_out /= jnp.linalg.norm(item_out, axis=1).reshape(-1, 1) #[n_item, dim] norm L2
        iid_feat = item_out[iid] #[bs, dim]
        pid_feat = item_out[pid] #[bs, dim]
        nid_feat = item_out[nid]

        pos_score = jnp.sum(iid_feat * pid_feat, axis=1)
        neg_score = jnp.sum(iid_feat * nid_feat, axis=1)  # [bs]

        loss1 = -jnp.log(nn.sigmoid(pos_score - neg_score))
        loss1 = loss1.mean()

        loss2 = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=target).mean()
        
        loss = loss1 + loss2
        return loss, {"loss1": loss1, "loss2": loss2, "item_feat": item_out}

    aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch, target, iidata)
    state = state.apply_gradients(grads=grads)
    loss, aux_dict = aux
    return state, loss, aux_dict


def train(state, dataloader, epochs, device):
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            batch, target, iidata = data
            batch = jnp.array(batch)
            target = jnp.array(target)
            iidata = jnp.array(iidata)
            state, loss, aux_dict = jax.jit(train_step, device=device)(state, batch, target, iidata)
            item_feat = aux_dict["item_feat"]
            loss1 = aux_dict["loss1"]
            loss2 = aux_dict["loss2"]
            dataloader.dataset.item_feat = item_feat # update item feat
            pbar.set_description("epoch: %i loss_C: %.4f loss_E: %.4f" % (epoch, loss1, loss2))
    return state


def eval(state, test_dataloader):
    print("EVALUATING")
    cnt = 0
    all_genbundles = []
    for test_data in test_dataloader:
        inp, bundles = test_data
        # print(inp, bundles)
        # exit()
        # inp = jnp.array([inp])
        inp = jnp.array(inp)
        gen_bundles = state.apply_fn(state.params, X=inp, test=True)
        all_genbundles.append(gen_bundles)
    all_genbundles = np.concatenate(all_genbundles, axis=0)
    return cnt, all_genbundles
    #     all_genbundles.extend(gen_bundles)
    #     print("gen buns:", gen_bundles)
    #     print("grd_buns:", bundles)
    #     for gd_bun in bundles:
    #         for gen_bundle in gen_bundles:
    #             if np.array_equal(gen_bundle, gd_bun):
    #                 cnt += 1
    # print(f"exactly hit: {cnt}")
    # return cnt, all_genbundles


def main():
    """
    Load Config & Init
    """
    args = get_args()
    dataset_name = args.dataset
    conf["dataset"] = args.dataset
    conf["data_path"] = args.data_path
    nu, nb, ni = get_size(f"{conf['data_path']}/{dataset_name}/{dataset_name}_data_size.txt")
    conf["n_user"] = nu
    conf["n_item"] = ni
    conf["n_bundle"] = nb
    conf["n_token"] = ni + 3
    conf["batch_size"] = args.batch_size
    conf["beam_width"] = args.beam_width
    conf["seq_len"] = args.seq_len
    conf["n_head"] = args.n_head
    conf["n_layer"] = args.n_layer
    conf["n_dim"] = args.n_dim
    conf["epochs"] = args.epochs
    # conf["p_epochs"] = args.p_epochs
    conf["continue_training"] = args.continue_training
    conf["weight_path"] = args.weight_path
    rng_gen, rng_model = jax.random.split(jax.random.PRNGKey(2024), num=2)
    np.random.seed(2024)

    devices = jax.devices()
    device = devices[args.device_id]
    conf["device"] = device

    print(conf)

    """
    Construct Training/Validating/Testing Data
    """
    if conf["dataset"] == "meal":
        train_data = TrainDataVer10(conf)
    else: # Steam Amazon meal
        train_data = TrainDataVer9(conf)
        
    # valid_data = TestDataVer2(conf, "valid") # I comment valid in order to get test-result faster
    if conf["dataset"] == "meal":
        test_data = TestDataVer3(conf, "test")
    else:
        test_data = TestDataVer2(conf, "test")
    """
    Train Main Model
    """
    sample_placeholder = jnp.zeros([2, conf["batch_size"],
                                    conf["seq_len"]], dtype=jnp.int32)

    model = BRIDGE(conf, train_data.ui_graph)
    print(f"MODEL NAME: {model.__class__.__name__}")
    print(f"DATACLASS: {train_data.__class__.__name__}, {test_data.__class__.__name__}")
    params = model.init(rng_model, sample_placeholder)

    # continue training with saved params
    if conf["continue_training"]:
        with open(conf["weight_path"], 'rb') as f:
            params = pickle.load(f)

    optimizer = optax.adam(learning_rate=1e-3)
    dataloader = DataLoader(train_data,
                            batch_size=conf["batch_size"],
                            # shuffle=True,
                            drop_last=False)

    state = train_state.TrainState.create(apply_fn=model.apply,
                                          params=params,
                                          tx=optimizer)
    """
    Training & Save checkpoint
    """
    logits, item_feat_out = state.apply_fn(state.params, sample_placeholder)
    dataloader.dataset.item_feat = item_feat_out
    state = train(state, dataloader, conf["epochs"], device)
    logits, item_feat_out = state.apply_fn(state.params, sample_placeholder)
    np.save(str(os.path.join(conf["data_path"], conf["dataset"], "item_feat.npy")), item_feat_out)

    with open(os.path.join(conf["data_path"], conf["dataset"], "main_model.pkl"), "wb") as f:
        pickle.dump(state.params, f)
    """
    Generate & Evaluate
    """
    test_dataloader = DataLoader(test_data, 
                                 batch_size=conf["batch_size"], 
                                 drop_last=False)
    # cnt_valid, generated_bundes_valid = eval(state, valid_data)
    cnt_test, generated_bundles_test = eval(state, test_dataloader)

    # with open(os.path.join(conf["data_path"], conf["dataset"], "generated_bundles_list_valid.pkl"), "wb") as f:
        # pickle.dump(generated_bundes_valid, f)
    with open(os.path.join(conf["data_path"], conf["dataset"], "generated_bundles_list_test.pkl"), "wb") as f:
        pickle.dump(generated_bundles_test, f)


if __name__ == "__main__":
    main()
