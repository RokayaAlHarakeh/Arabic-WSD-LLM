import io
p = 'make_figures.py'
s = io.open(p, encoding='utf-8').read()
start = s.index('def fig_attention():')
end   = s.index('def fig_prompt_flow():')
new = '''def fig_attention():
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    toks = ["he", "escaped", "the", "party", "by", "sleeping"]
    xs = np.linspace(0.45, 8.35, len(toks))
    for x, t in zip(xs, toks):
        fc = "#EEF3F8" if t == "escaped" else "white"
        ec = RED if t == "escaped" else GREY
        _box(ax, x, 0.45, 1.30, 0.62, t, fc=fc, ec=ec, fs=8.5,
             bold=(t == "escaped"), lw=(1.6 if t == "escaped" else 1.0))
    tgt_x, tgt_y = 2.90, 2.55
    weights = [0.05, 0.0, 0.06, 0.34, 0.10, 0.45]
    for x, w in zip(xs, weights):
        if w <= 0:
            continue
        # straight lines only: nothing routes through a neighbouring box
        ax.add_patch(FancyArrowPatch((x + 0.65, 1.07), (tgt_x, tgt_y), arrowstyle="-|>",
                                     mutation_scale=9, color=RED, lw=0.6 + 5.0 * w,
                                     alpha=0.30 + 0.60 * w))
    _box(ax, tgt_x - 1.55, tgt_y, 3.1, 0.85,
         "attention-weighted\nrepresentation of \\"escaped\\"", fc="#EEF3F8", ec=RED, fs=8)
    ax.text(6.9, 3.05, r"$\mathrm{softmax}\!\left(\dfrac{QK^{\top}}{\sqrt{d_k}}\right)V$",
            fontsize=12, ha="center", va="center")
    ax.text(6.9, 2.30, "arrow thickness = attention weight", fontsize=7.5,
            ha="center", color=GREY)
    ax.text(5.0, 4.85, "\\"sleeping\\" and \\"party\\" carry most of the weight —\n"
                       "which is what selects the figurative sense of the verb",
            fontsize=8.5, ha="center", color=NAVY, linespacing=1.5)
    ax.set_title("Self-attention conditions a token on its context", fontsize=11)
    save(fig, "fig_attention.png")


'''
io.open(p, 'w', encoding='utf-8').write(s[:start] + new + s[end:])
import ast; ast.parse(io.open(p, encoding='utf-8').read())
print('fig_attention rewritten; syntax OK')
