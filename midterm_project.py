"""
CMPSC 448 midterm: Who Wrote It? Identifying LLMs from Their Responses
Tokenizer, CNN + BiLSTM models, training, RQ1 + RQ2 experiments, and figures.

Input: data/dataset.csv (included with the project). It holds 9,636 rows of
(LLM_name, family, LLM_input, LLM_output, source_subset, prompt_id, task) built from the public
AlpacaEval model outputs; see the report, Section 1, for where the data came from.

Outputs are written next to this file: results/, figures/.
"""
import argparse, json, os, re, time
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT, "data", "dataset.csv")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
SEEDS = [0, 1, 2]
EPOCHS=20
SIM_REPS=5
torch.set_num_threads(max(1, os.cpu_count() or 1))


def load():
    if not os.path.exists(DATASET):
        raise SystemExit(f"{DATASET} not found. Put the project's dataset.csv in a data/ folder next to this script.")
    df = pd.read_csv(DATASET)
    classes = sorted(df.family.unique())
    return df, classes


def labels(d, classes, col="family"):
    m = {c: i for i, c in enumerate(classes)}
    return d[col].map(m).values


def save_json(obj, name):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1, default=conv)

_TOKEN_RE = re.compile(r"\n|[A-Za-z]+(?:'[a-z]+)?|\d+|[^\sA-Za-z\d]")

PAD, UNK, NL, SEP = "<pad>", "<unk>", "<nl>", "<sep>"


def tokenize(text: str):
    return [NL if t == "\n" else t for t in _TOKEN_RE.findall(text)]


def build_vocab(token_lists, min_freq=2, max_size=30000):
    counts = Counter(t for toks in token_lists for t in toks)
    itos = [PAD, UNK, NL, SEP] + [t for t, c in counts.most_common(max_size) if c >= min_freq and t not in (NL, SEP)]
    return {t: i for i, t in enumerate(itos)}


def encode(tokens, vocab, max_len):
    ids = [vocab.get(t, 1) for t in tokens[:max_len]]
    return ids if ids else [1]


def make_tokens(df, mode, max_len=400, max_prompt=120):
    """mode: 'output' | 'input' | 'both'. For 'both' the prompt is truncated to max_prompt
    tokens, followed by <sep> and the response, so the response always gets most of the budget."""
    out = []
    for inp, resp in zip(df["LLM_input"], df["LLM_output"]):
        if mode == "output":
            toks = tokenize(resp)
        elif mode == "input":
            toks = tokenize(inp)
        else:
            toks = tokenize(inp)[:max_prompt] + [SEP] + tokenize(resp)
        out.append(toks[:max_len])
    return out


def group_split(df, seed=0, frac=(0.7, 0.15, 0.15)):
    """Split by prompt so that no prompt (and none of its answers) appears in two splits."""
    rng = np.random.RandomState(seed)
    pids = np.array(sorted(df["prompt_id"].unique()))
    rng.shuffle(pids)
    n_tr, n_va = int(frac[0] * len(pids)), int(frac[1] * len(pids))
    tr, va = set(pids[:n_tr]), set(pids[n_tr:n_tr + n_va])
    split = df["prompt_id"].map(lambda p: "train" if p in tr else ("val" if p in va else "test"))
    return df[split == "train"], df[split == "val"], df[split == "test"]

class TextCNN(nn.Module):
    """Embedding -> parallel 1-D convolutions (widths 3/4/5) -> global max-pool -> linear.
    Each filter acts as a learned n-gram detector; max-pooling asks "does this pattern
    occur anywhere in the response?", which suits stylistic fingerprints."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128, widths=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, w, padding=w // 2) for w in widths])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(widths), n_classes)

    def forward(self, x, lengths):
        e = self.emb(x).transpose(1, 2)
        mask = (x != 0).unsqueeze(1)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(e))[:, :, : x.size(1)]
            h = h.masked_fill(~mask, -1e4)
            pooled.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """Embedding -> 1-layer bidirectional LSTM -> concat of masked mean-pool and max-pool
    over time -> linear. Batches are length-bucketed (see train._batches), so padding is
    only a few tokens; we skip pack_padded_sequence because its backward pass was ~10x
    slower on CPU in our timing tests."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        h, _ = self.lstm(e)
        mask = (x != 0).unsqueeze(2).float()
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = h.masked_fill(mask == 0, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1)))





def _batches(ids, y, bs, shuffle, rng):
    """Length-bucketed batches: examples of similar length go together, which keeps
    padding (and the cost of packed LSTM sequences on CPU) low."""
    if shuffle:
        order = rng.permutation(len(ids))
        chunks = [sorted(order[i:i + bs * 50], key=lambda j: len(ids[j])) for i in range(0, len(order), bs * 50)]
        batches = [c[i:i + bs] for c in chunks for i in range(0, len(c), bs)]
        batches = [batches[k] for k in rng.permutation(len(batches))]
    else:
        order = sorted(range(len(ids)), key=lambda j: len(ids[j]))
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
    for idx in batches:
        idx = np.asarray(idx)
        lens = torch.tensor([len(ids[j]) for j in idx])
        x = torch.zeros(len(idx), int(lens.max()), dtype=torch.long)
        for k, j in enumerate(idx):
            x[k, : lens[k]] = torch.tensor(ids[j])
        yield x, lens, torch.tensor(y[idx]), idx


def _predict(model, ids, y, bs=128):
    model.eval()
    preds = np.zeros(len(ids), dtype=int)
    with torch.no_grad():
        for x, lens, _, idx in _batches(ids, y, bs, False, None):
            preds[idx] = model(x, lens).argmax(1).numpy()
    return preds


def run_nn(arch, train_toks, y_tr, val_toks, y_va, test_toks, y_te, n_classes, seed=0,
           max_len=400, epochs=15, patience=3, lr=1e-3, bs=64, verbose=True):
    """Train with Adam + cross-entropy, early-stop on validation macro-F1, restore the best
    checkpoint, and evaluate once on the test split."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    vocab = build_vocab(train_toks)
    enc = lambda T: [encode(t, vocab, max_len) for t in T]
    tr, va, te = enc(train_toks), enc(val_toks), enc(test_toks)
    Model = {"cnn": TextCNN, "lstm": BiLSTM}[arch]
    model = Model(len(vocab), n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, bad, history = -1, None, 0, []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, lens, yb, _ in _batches(tr, y_tr, bs, True, rng):
            opt.zero_grad()
            loss = loss_fn(model(x, lens), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(yb)
        vp = _predict(model, va, y_va)
        vf1 = f1_score(y_va, vp, average="macro")
        history.append(dict(epoch=ep + 1, train_loss=tot / len(tr), val_acc=accuracy_score(y_va, vp), val_f1=vf1))
        if verbose:
            print(f"  [{arch}] ep{ep+1:2d} loss={tot/len(tr):.3f} val_acc={history[-1]['val_acc']:.3f} val_f1={vf1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best:
            best, bad = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    tp = _predict(model, te, y_te)
    return dict(acc=accuracy_score(y_te, tp), f1=f1_score(y_te, tp, average="macro"), pred=tp,
                cm=confusion_matrix(y_te, tp, labels=range(n_classes)), history=history,
                n_params=sum(p.numel() for p in model.parameters()), vocab_size=len(vocab), model=model, vocab=vocab)


def run_tfidf(train_texts, y_tr, test_texts, y_te, n_classes, C=10.0):
    """Non-neural reference: word 1-2-gram + char 2-5-gram TF-IDF into logistic regression."""
    vec = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True, lowercase=False, token_pattern=r"\S+"),
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=150000, sublinear_tf=True, lowercase=False),
    )
    clf = make_pipeline(vec, LogisticRegression(C=C, max_iter=2000))
    clf.fit(train_texts, y_tr)
    tp = clf.predict(test_texts)
    return dict(acc=accuracy_score(y_te, tp), f1=f1_score(y_te, tp, average="macro"), pred=tp,
                cm=confusion_matrix(y_te, tp, labels=range(n_classes)), clf=clf)


def rq1_rq2():
    """RQ1 (output-only CNN / BiLSTM) and RQ2 (input-only vs output-only vs input+output).
    Same prompt-grouped 70/15/15 split for every run; several random initialisations per model."""
    df, classes = load()
    tr, va, te = group_split(df, seed=0)
    y_tr, y_va, y_te = (labels(d, classes) for d in (tr, va, te))
    print("split sizes", len(tr), len(va), len(te), "prompts", tr.prompt_id.nunique(), va.prompt_id.nunique(), te.prompt_id.nunique())

    results, preds = [], te[["LLM_name", "family", "task", "prompt_id"]].copy()


    def raw_text(d, mode):
        if mode == "output":
            return d.LLM_output.tolist()
        if mode == "input":
            return d.LLM_input.tolist()
        return (d.LLM_input + f" {SEP} " + d.LLM_output).tolist()


    for mode in ["output", "input", "both"]:
        toks = [make_tokens(d, mode) for d in (tr, va, te)]
        r = run_tfidf(raw_text(tr, mode), y_tr, raw_text(te, mode), y_te, len(classes))
        results.append(dict(mode=mode, model="tfidf_lr", seed=0, acc=r["acc"], f1=r["f1"], cm=r["cm"]))
        preds[f"tfidf_{mode}"] = r["pred"]
        print(mode, "tfidf", round(r["acc"], 3), flush=True)
        for arch in ["cnn", "lstm"]:
            for seed in SEEDS:
                r = run_nn(arch, toks[0], y_tr, toks[1], y_va, toks[2], y_te, len(classes), seed=seed, epochs=EPOCHS)
                results.append(dict(mode=mode, model=arch, seed=seed, acc=r["acc"], f1=r["f1"], cm=r["cm"],
                                    history=r["history"], n_params=r["n_params"], vocab_size=r["vocab_size"]))
                preds[f"{arch}_{mode}_s{seed}"] = r["pred"]
                print(mode, arch, seed, round(r["acc"], 3), round(r["f1"], 3), flush=True)
                save_json(results, "rq1_rq2.json")
                preds.to_csv(os.path.join(RESULTS, "rq1_rq2_test_predictions.csv"), index=False)


    maj = np.bincount(y_tr).argmax()
    results.append(dict(mode="-", model="majority", seed=0, acc=float((y_te == maj).mean()), f1=None))
    save_json(results, "rq1_rq2.json")


    models = sorted(df.LLM_name.unique())
    toks = [make_tokens(d, "output") for d in (tr, va, te)]
    r = run_nn("cnn", toks[0], labels(tr, models, "LLM_name"), toks[1], labels(va, models, "LLM_name"),
               toks[2], labels(te, models, "LLM_name"), len(models), seed=0, epochs=EPOCHS)
    save_json(dict(models=models, acc=r["acc"], f1=r["f1"], cm=r["cm"]), "rq1_model_level_cnn.json")
    print("model-level cnn", r["acc"], r["f1"])



def rq2_sim():
    """When can the prompt alone reveal the LLM? Each prompt is given to exactly one family,
    chosen with probability `bias` by the prompt's source subset (else at random)."""
    df, classes = load()
    HOME = {"selfinstruct": "GPT", "oasst": "Llama", "koala": "Claude", "helpful_base": "Gemini/Gemma", "vicuna": "GPT"}
    prompts = df.drop_duplicates("prompt_id")[["prompt_id", "source_subset"]].reset_index(drop=True)
    out = json.load(open(f"{RESULTS}/rq2_sim.json")) if os.path.exists(f"{RESULTS}/rq2_sim.json") else []   # resume
    done = {(r["bias"], r["rep"]) for r in out}
    for bias in [0.0, 0.3, 0.6, 0.9]:
        for rep in range(SIM_REPS):
            if (bias, rep) in done:
                continue
            rng = np.random.RandomState(100 * rep + int(bias * 10))
            fam = [HOME[s] if rng.rand() < bias else classes[rng.randint(len(classes))] for s in prompts.source_subset]
            chosen = []
            for pid, f in zip(prompts.prompt_id, fam):
                cand = df[(df.prompt_id == pid) & (df.family == f)]
                if len(cand):
                    chosen.append(cand.sample(1, random_state=rng).index[0])
            sim = df.loc[chosen]
            perm = rng.permutation(len(sim))
            n_tr, n_va = int(0.7 * len(sim)), int(0.15 * len(sim))
            tr, va, te = sim.iloc[perm[:n_tr]], sim.iloc[perm[n_tr:n_tr + n_va]], sim.iloc[perm[n_tr + n_va:]]
            y = [labels(d, classes) for d in (tr, va, te)]
            row = dict(bias=bias, rep=rep, n=len(sim), majority=float((y[2] == np.bincount(y[0], minlength=len(classes)).argmax()).mean()))
            for mode, col in [("input", "LLM_input"), ("output", "LLM_output")]:
                r = run_tfidf(pd.concat([tr, va])[col].tolist(), np.concatenate([y[0], y[1]]), te[col].tolist(), y[2], len(classes))
                row[f"tfidf_{mode}"] = r["acc"]
                toks = [make_tokens(d, mode) for d in (tr, va, te)]
                r = run_nn("cnn", toks[0], y[0], toks[1], y[1], toks[2], y[2], len(classes), seed=rep, epochs=EPOCHS, verbose=False)
                row[f"cnn_{mode}"] = r["acc"]
            print(row, flush=True)
            out.append(row)
            save_json(out, "rq2_sim.json")



FAM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order = sorted family names
MODEL_COLORS = {"majority": "#b4b2a9", "tfidf_lr": "#7c7a72", "cnn": "#2a78d6", "lstm": "#eb6834"}
MODEL_NAMES = {"majority": "Majority", "tfidf_lr": "TF-IDF + LR", "cnn": "CNN", "lstm": "BiLSTM"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.edgecolor": "#8a8980", "savefig.dpi": 200, "savefig.bbox": "tight"})

def make_figures():
    df, classes = load()
    K = len(classes)
    os.makedirs(FIGS, exist_ok=True)
    tables = {}
    def jl(name):
        p = os.path.join(RESULTS, name)
        return json.load(open(p)) if os.path.exists(p) else None


    def bar_labels(ax, bars, fmt="{:.0%}"):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, fmt.format(b.get_height()), ha="center", va="bottom", fontsize=8, color="#52514e")

    per_model = df.groupby(["family", "LLM_name"]).size().rename("n").reset_index()
    tables["dataset_per_model"] = per_model
    tables["tasks"] = df.drop_duplicates("prompt_id").task.value_counts()
    tables["subsets"] = df.drop_duplicates("prompt_id").source_subset.value_counts()
    words = df.LLM_output.str.split().str.len()
    tables["len_by_family"] = words.groupby(df.family).describe()[["mean", "25%", "50%", "75%"]].round(0)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    data = [words[df.family == c].clip(upper=1200) for c in classes]
    bp = ax.boxplot(data, orientation="horizontal", tick_labels=classes, showfliers=False, patch_artist=True, widths=0.6, medianprops=dict(color="#0b0b0b"))
    for p, c in zip(bp["boxes"], FAM_COLORS):
        p.set_facecolor(c); p.set_alpha(0.85); p.set_edgecolor("white")
    ax.set_xlabel("Response length (words)"); ax.set_title("Response length by LLM family", loc="left", fontsize=11)
    fig.savefig(f"{FIGS}/dataset_lengths.png"); plt.close(fig)


    r12 = jl("rq1_rq2.json")
    if r12:
        t = pd.DataFrame(r12)
        summ = t.groupby(["mode", "model"]).agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"), f1_mean=("f1", "mean"), f1_std=("f1", "std"), runs=("acc", "size")).reset_index()
        tables["rq1_rq2_summary"] = summ
        # RQ1 bar chart
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        order = ["majority", "tfidf_lr", "cnn", "lstm"]
        vals, errs = [], []
        for m in order:
            row = summ[(summ.model == m) & (summ["mode"].isin(["output", "-"]))]
            vals.append(row.acc_mean.iloc[0] if len(row) else np.nan); errs.append(np.nan_to_num(row.acc_std.iloc[0]) if len(row) else 0)
        bars = ax.bar([MODEL_NAMES[m] for m in order], vals, yerr=errs, color=[MODEL_COLORS[m] for m in order], width=0.6, capsize=3, error_kw=dict(ecolor="#52514e", lw=1))
        bar_labels(ax, bars); ax.set_ylim(0, 1); ax.set_ylabel("Test accuracy")
        ax.set_title(f"RQ1: {K}-way family identification from the response", loc="left", fontsize=11)
        fig.savefig(f"{FIGS}/rq1_accuracy.png"); plt.close(fig)
        # confusion matrices (seed 0, output mode)
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
        for ax, m in zip(axes, ["cnn", "lstm"]):
            rows = [r for r in r12 if r["model"] == m and r["mode"] == "output" and r["seed"] == 0]
            if not rows: continue
            cm = np.array(rows[0]["cm"], dtype=float); cmn = cm / cm.sum(1, keepdims=True)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
            ax.set_xticks(range(K)); ax.set_xticklabels(classes, rotation=40, ha="right"); ax.set_yticks(range(K)); ax.set_yticklabels(classes)
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.55 else "#0b0b0b")
            ax.set_title(f"{MODEL_NAMES[m]} (row-normalised)", fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.tight_layout(); fig.savefig(f"{FIGS}/rq1_confusion.png"); plt.close(fig)
        # training curves
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for m in ["cnn", "lstm"]:
            for r in r12:
                if r["model"] == m and r["mode"] == "output":
                    h = pd.DataFrame(r["history"])
                    axes[0].plot(h.epoch, h.train_loss, color=MODEL_COLORS[m], lw=1.5, alpha=0.8, label=MODEL_NAMES[m] if r["seed"] == 0 else None)
                    axes[1].plot(h.epoch, h.val_acc, color=MODEL_COLORS[m], lw=1.5, alpha=0.8, label=MODEL_NAMES[m] if r["seed"] == 0 else None)
        axes[0].set_title("Training loss", fontsize=10); axes[1].set_title("Validation accuracy", fontsize=10)
        for a in axes: a.set_xlabel("Epoch"); a.legend(frameon=False)
        fig.tight_layout(); fig.savefig(f"{FIGS}/training_curves.png"); plt.close(fig)
        # per-model accuracy of CNN seed0 on output
        pr = pd.read_csv(f"{RESULTS}/rq1_rq2_test_predictions.csv")
        fam_idx = pr.family.map({c: i for i, c in enumerate(classes)})
        per = {}
        for col in [c for c in pr.columns if c.endswith("_output_s0") or c == "tfidf_output"]:
            per[col] = (pr[col] == fam_idx).groupby(pr.LLM_name).mean()
        tables["rq1_per_model"] = pd.DataFrame(per).round(3)
        tables["rq1_per_task"] = pd.DataFrame({col: (pr[col] == fam_idx).groupby(pr.task).mean() for col in per}).round(3)
        # RQ2 grouped bars
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        modes = ["input", "output", "both"]; mlabels = ["Input only", "Output only", "Input + output"]
        w = 0.25
        for k, m in enumerate(["tfidf_lr", "cnn", "lstm"]):
            v = [summ[(summ.model == m) & (summ["mode"] == md)].acc_mean.iloc[0] if len(summ[(summ.model == m) & (summ["mode"] == md)]) else np.nan for md in modes]
            e = [np.nan_to_num(summ[(summ.model == m) & (summ["mode"] == md)].acc_std.iloc[0]) if len(summ[(summ.model == m) & (summ["mode"] == md)]) else 0 for md in modes]
            bars = ax.bar(np.arange(3) + (k - 1) * w, v, w * 0.92, yerr=e, color=MODEL_COLORS[m], label=MODEL_NAMES[m], capsize=2, error_kw=dict(ecolor="#52514e", lw=1))
            bar_labels(ax, bars)
        ax.axhline(1 / K, color="#52514e", ls="--", lw=1); ax.text(2.45, 1 / K + 0.015, "chance", fontsize=8, color="#52514e", ha="right")
        ax.set_xticks(range(3)); ax.set_xticklabels(mlabels); ax.set_ylim(0, 1); ax.set_ylabel("Test accuracy"); ax.legend(frameon=False, ncol=3, loc="upper left")
        ax.set_title("RQ2: what the classifier sees", loc="left", fontsize=11)
        fig.savefig(f"{FIGS}/rq2_modes.png"); plt.close(fig)

    rml = jl("rq1_model_level_cnn.json")
    if rml:
        tables["rq1_model_level"] = pd.Series(dict(acc=rml["acc"], f1=rml["f1"]))
        cm = np.array(rml["cm"], float); cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        ax.set_xticks(range(len(rml['models']))); ax.set_xticklabels(rml["models"], rotation=70, ha="right", fontsize=7); ax.set_yticks(range(len(rml['models']))); ax.set_yticklabels(rml["models"], fontsize=7)
        ax.set_title(f"CNN, {len(rml['models'])}-way model identification (acc {rml['acc']:.1%})", fontsize=10)
        fig.savefig(f"{FIGS}/rq1_model_level_confusion.png"); plt.close(fig)

    sim = jl("rq2_sim.json")
    if sim:
        s = pd.DataFrame(sim).groupby("bias").agg(["mean", "std"])
        tables["rq2_sim"] = s.round(3)
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        for col, lab, c, ls in [("tfidf_input", "TF-IDF, input only", "#7c7a72", "-"), ("cnn_input", "CNN, input only", "#2a78d6", "-"),
                                ("tfidf_output", "TF-IDF, output only", "#7c7a72", ":"), ("cnn_output", "CNN, output only", "#2a78d6", ":")]:
            ax.errorbar(s.index, s[(col, "mean")], yerr=s[(col, "std")], color=c, ls=ls, lw=2, marker="o", ms=5, capsize=2, label=lab)
        ax.axhline(1 / K, color="#52514e", ls="--", lw=1)
        ax.set_xlabel("Collection bias (share of prompts routed by source)"); ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8); ax.set_title("RQ2: prompt-only accuracy under biased collection", loc="left", fontsize=11)
        fig.savefig(f"{FIGS}/rq2_sim.png"); plt.close(fig)

    with open(f"{RESULTS}/tables.txt", "w") as f:
        for k, v in tables.items():
            f.write(f"===== {k}\n{v.to_string() if hasattr(v, 'to_string') else v}\n\n")
    print("figures ->", FIGS, "| tables ->", os.path.join(RESULTS, "tables.txt"))



#main
STEPS = {"rq1_rq2": rq1_rq2,
         "rq2_sim": rq2_sim, "figures": make_figures}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS), help="run only this step (default: all, in order)")
    ap.add_argument("--quick", action="store_true", help="fast smoke test: 1 seed, 2 epochs, 1 sim repeat")
    args = ap.parse_args()
    if args.quick:
        SEEDS, EPOCHS, SIM_REPS = [0], 2, 1
        RESULTS = os.path.join(ROOT, "results_quick")
        FIGS = os.path.join(ROOT, "figures_quick")
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    for name, fn in STEPS.items():
        if args.step in (None, name):
            print(f"\n===== {name} =====", flush=True)
            fn()
