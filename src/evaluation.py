import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_eer(genuine_scores, impostor_scores):
    genuine  = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    thresholds = np.sort(np.concatenate([genuine, impostor]))

    best_thresh, best_eer, min_diff = thresholds[0], 1.0, float('inf')
    for t in thresholds:
        fmr  = np.mean(impostor <= t)
        fnmr = np.mean(genuine  >  t)
        diff = abs(fmr - fnmr)
        if diff < min_diff:
            min_diff    = diff
            best_thresh = t
            best_eer    = (fmr + fnmr) / 2.0
    return best_eer, best_thresh


def compute_dprime(genuine_scores, impostor_scores):
    genuine  = np.array(genuine_scores, dtype=float)
    impostor = np.array(impostor_scores, dtype=float)
    mu_g, sig_g = genuine.mean(),  genuine.std()
    mu_i, sig_i = impostor.mean(), impostor.std()
    return abs(mu_i - mu_g) / np.sqrt(0.5 * (sig_g**2 + sig_i**2) + 1e-10)


def compute_roc(genuine_scores, impostor_scores, n_points=500):
    genuine  = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    all_sc   = np.sort(np.concatenate([genuine, impostor]))
    thresholds = np.linspace(all_sc.min(), all_sc.max(), n_points)
    fmr_list, tmr_list = [], []
    for t in thresholds:
        fmr_list.append(np.mean(impostor <= t))
        tmr_list.append(np.mean(genuine  <= t))
    return np.array(fmr_list), np.array(tmr_list)


def compute_tmr_at_fmr(genuine_scores, impostor_scores, fmr_target):
    genuine  = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    all_sc   = np.sort(np.concatenate([genuine, impostor]))
    best_tmr = 0.0
    for t in all_sc:
        if np.mean(impostor <= t) <= fmr_target:
            tmr = np.mean(genuine <= t)
            if tmr > best_tmr:
                best_tmr = tmr
    return best_tmr


def fuse_genuine_impostor(genuine_lists, impostor_lists):
    fused_gen_parts, fused_imp_parts = [], []
    for gen, imp in zip(genuine_lists, impostor_lists):
        gen = np.array(gen, dtype=float)
        imp = np.array(imp, dtype=float)
        combined = np.concatenate([gen, imp])
        lo, hi   = combined.min(), combined.max()
        span     = hi - lo if hi - lo > 1e-10 else 1.0
        fused_gen_parts.append((gen - lo) / span)
        fused_imp_parts.append((imp - lo) / span)
    return np.mean(fused_gen_parts, axis=0), np.mean(fused_imp_parts, axis=0)


def collect_metrics(label, genuine_scores, impostor_scores, rank1_accuracy):
    eer, _  = compute_eer(genuine_scores, impostor_scores)
    dprime  = compute_dprime(genuine_scores, impostor_scores)
    tmr_1   = compute_tmr_at_fmr(genuine_scores, impostor_scores, fmr_target=0.01)
    tmr_001 = compute_tmr_at_fmr(genuine_scores, impostor_scores, fmr_target=0.0001)
    return {
        'label'  : label,
        'rank1'  : rank1_accuracy,
        'eer'    : eer * 100,
        'dprime' : dprime,
        'tmr_1'  : tmr_1 * 100,
        'tmr_001': tmr_001 * 100,
    }


def print_results_table(all_metrics):
 
    lbp   = all_metrics[0]
    eig   = all_metrics[1]
    hog   = all_metrics[2]
    fused = all_metrics[3]
 
    print('\nTABLE — Biometric Evaluation Results')
    print('-' * 90)
    print(f"{'Metric':<25} {'LBP (A)':<20} {'Eigenfaces (B)':<20} {'HOG (C)':<20} {'Fused (A+B+C)':<20}")
    print('-' * 90)
    print(f"{'EER (%)':<25} {lbp['eer']:<20.2f} {eig['eer']:<20.2f} {hog['eer']:<20.2f} {fused['eer']:<20.2f}")
    print(f"{'D-prime':<25} {lbp['dprime']:<20.4f} {eig['dprime']:<20.4f} {hog['dprime']:<20.4f} {fused['dprime']:<20.4f}")
    print(f"{'TMR @ FMR = 1%':<25} {lbp['tmr_1']:<20.2f} {eig['tmr_1']:<20.2f} {hog['tmr_1']:<20.2f} {fused['tmr_1']:<20.2f}")
    print(f"{'TMR @ FMR = 0.01%':<25} {lbp['tmr_001']:<20.2f} {eig['tmr_001']:<20.2f} {hog['tmr_001']:<20.2f} {fused['tmr_001']:<20.2f}")
    print(f"{'Rank-1 Accuracy (%)':<25} {lbp['rank1']:<20.2f} {eig['rank1']:<20.2f} {hog['rank1']:<20.2f} {fused['rank1']:<20.2f}")
    print('-' * 90)


def plot_score_distribution(genuine_scores, impostor_scores, method_name, save_path=None):
    gen = np.array(genuine_scores)
    imp = np.array(impostor_scores)
    eer, eer_thresh = compute_eer(genuine_scores, impostor_scores)
    dprime = compute_dprime(genuine_scores, impostor_scores)

    fig, ax = plt.subplots(figsize=(8, 4))
    all_vals = np.concatenate([gen, imp])
    edges = np.linspace(all_vals.min(), all_vals.max(), 41)

    ax.hist(gen, bins=edges, density=True, alpha=0.6,
            color='#2196F3', label='Genuine', edgecolor='none')
    ax.hist(imp, bins=edges, density=True, alpha=0.5,
            color='#F44336', label='Impostor', edgecolor='none')
    ax.axvline(eer_thresh, color='black', linestyle='--', linewidth=1.2,
               label=f'EER threshold = {eer_thresh:.3f}')

    ax.set_xlabel('Distance score (lower = more similar)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(
        f'Genuine vs Impostor — {method_name}\n'
        f"EER = {eer*100:.2f}%   |   D\u2032 = {dprime:.2f}",
        fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_roc_comparison(methods_data, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in methods_data:
        fmr, tmr = compute_roc(m['genuine'], m['impostor'], n_points=1000)
        ax.semilogx(np.clip(fmr, 1e-5, 1.0), tmr,
                    label=m['label'],
                    color=m.get('color'),
                    linestyle=m.get('linestyle', '-'),
                    linewidth=2)

    for xv, lbl in [(0.01, 'FMR=1%'), (0.0001, 'FMR=0.01%')]:
        ax.axvline(xv, color='grey', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.text(xv * 1.1, 0.05, lbl, fontsize=8, color='grey')

    ax.set_xlabel('False Match Rate (FMR)', fontsize=11)
    ax.set_ylabel('True Match Rate (TMR)', fontsize=11)
    ax.set_title('ROC Curves — All Methods', fontsize=13)
    ax.set_xlim([1e-5, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_full_evaluation(lbp_genuine,   lbp_impostor,   lbp_rank1,
                        eig_genuine,   eig_impostor,   eig_rank1,
                        hog_genuine,   hog_impostor,   hog_rank1,
                        fused_genuine, fused_impostor, fused_rank1):
   
    lbp_m   = collect_metrics('LBP',          lbp_genuine,   lbp_impostor,   lbp_rank1)
    eig_m   = collect_metrics('Eigenfaces',    eig_genuine,   eig_impostor,   eig_rank1)
    hog_m   = collect_metrics('HOG',           hog_genuine,   hog_impostor,   hog_rank1)
    fused_m = collect_metrics('Fused (A+B+C)', fused_genuine, fused_impostor, fused_rank1)

    print_results_table([lbp_m, eig_m, hog_m, fused_m])

    plot_score_distribution(lbp_genuine,   lbp_impostor,   'LBP',
                            save_path='gen_imp_LBP.png')
    plot_score_distribution(eig_genuine,   eig_impostor,   'Eigenfaces',
                            save_path='gen_imp_Eigenfaces.png')
    plot_score_distribution(hog_genuine,   hog_impostor,   'HOG',
                            save_path='gen_imp_HOG.png')
    plot_score_distribution(fused_genuine, fused_impostor, 'Fused (A+B+C)',
                            save_path='gen_imp_Fused.png')

    plot_roc_comparison([
        {'label': 'LBP',          'genuine': lbp_genuine,   'impostor': lbp_impostor,
         'color': 'Red', 'linestyle': '--'},
        {'label': 'Eigenfaces',   'genuine': eig_genuine,   'impostor': eig_impostor,
         'color': 'Blue', 'linestyle': '-.'},
        {'label': 'HOG',          'genuine': hog_genuine,   'impostor': hog_impostor,
         'color': 'Green', 'linestyle': ':'},
        {'label': 'Fused (A+B+C)','genuine': fused_genuine,'impostor': fused_impostor,
         'color': 'Purple', 'linestyle': '-'},
    ], save_path='roc_comparison.png')

    return [lbp_m, eig_m, hog_m, fused_m]