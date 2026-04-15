# CALCULATE SSF, PLOT RDF AND SSF
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def lorch_window(r, R):
    # 减少有限截断造成的影响
    x = np.pi * r / R
    w = np.sinc(x / np.pi)
    return w


def trapz_int(x, y):
    return np.trapezoid(y, x)


def parse_rdf_vector_file(fname, lenperblock=8):
    """Read tmp.rdf (mode vector). Returns r (Å), dict of g_ab, dict of coord_ab."""
    data_blocks = []
    current_block = []
    nrows = None
    with open(fname, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                if current_block:
                    data_blocks.append((nrows, current_block))
                    current_block = []

                nrows = int(parts[1])

            elif len(parts) == lenperblock and parts[0].isdigit():
                row_data = list(map(float, parts))
                current_block.append(row_data)

    if current_block:
        data_blocks.append((nrows, current_block))

    all_data = []
    for nrow, block in data_blocks:
        cur_df = create_dataframe(block)
        if nrow != nrows:
            raise ValueError("Cannot do average rdf between different time periods")
        else:
            all_data.append(cur_df)
    grouped = pd.concat(all_data).groupby(level=0)
    mean_df = grouped.mean()
    std_df = grouped.std()

    mean_df.columns = [col + "_mean" for col in mean_df.columns]
    std_df.columns = [col + "_std" for col in std_df.columns]

    final_df = pd.concat([mean_df, std_df], axis=1)

    return final_df


def create_dataframe(block_data):
    columns = ["row", "r", "rdf_OO", "n_OO", "rdf_OH", "n_OH", "rdf_HH", "n_HH"]
    df = pd.DataFrame(block_data, columns=columns)
    df.drop(columns=["row"], inplace=True)

    df.set_index("r", inplace=True)

    return df


def compute_sk(r, g_r, rho, k_vals, window=False):
    R = r[-1]
    gr_minusl = g_r - 1.0
    if window:
        gr_minusl *= lorch_window(r, R)
    sk = []
    for k in k_vals:
        if k == 0:
            integrand = 4.0 * np.pi * rho * r**2 * gr_minusl
            sk.append(1.0 + trapz_int(r, integrand))
        else:
            integrand = 4.0 * np.pi * rho * r**2 * gr_minusl * np.sin(k * r) / (k * r)
            sk.append(1.0 + trapz_int(r, integrand))
    return np.array(sk)


def plot_with_error(x, y, yerr, ylabel):
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    plt.xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{ylabel.replace(' ', '_').replace('$','').replace('{','').replace('}','').lower()}.png",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute S(k) from RDF data.")
    parser.add_argument("rdf_file", help="Path to RDF data file")
    parser.add_argument("--rho", type=float, required=True, help="Number Density(Å^-3)")
    parser.add_argument("--kmax", type=float, default=20.0, help="Maximum k value (Å^-1)")
    parser.add_argument("--nk", type=int, default=200, help="Number for k Sampling")
    parser.add_argument("--window", action="store_true", help="Use the Lorch window")

    args = parser.parse_args()
    # rdf = parse_rdf_vector_file(args.rdf_file)
    rdf = pd.read_csv(args.rdf_file, index_col=0)
    k_vals = np.linspace(0.0, args.kmax, args.nk)
    sk_OO = compute_sk(rdf.index.values, rdf["rdf_OO_mean"], args.rho, k_vals, window=args.window)

    # plot_with_error(rdf.index.values, rdf["rdf_OO_mean"], rdf["rdf_OO_std"], ylabel=r"$g_{OO}(r)$")
    # plot_with_error(rdf.index.values, rdf["rdf_OH_mean"], rdf["rdf_OH_std"], ylabel=r"$g_{OH}(r)$")
    # plot_with_error(rdf.index.values, rdf["rdf_HH_mean"], rdf["rdf_HH_std"], ylabel=r"$g_{HH}(r)$")

    out = np.column_stack((k_vals, sk_OO))
    np.savetxt("quenching\S_OO.data", out, header="k(Å^-1) S_OO(k)")

    plt.plot(k_vals, sk_OO, label="S_OO(k)")
    plt.xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)")
    plt.ylabel(r"$S(k)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("quenching\S_OO.png", dpi=300)
    plt.show()
