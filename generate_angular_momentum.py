"""Generate MESA angular momentum profile assuming solid body rotation.

Usage:
    uv run generate_angular_momentum.py --model models/adjust_He_envelope.mod \
        --J_CO 8.597e49 --J_total 8.597e49 --fout data_J_adjust_He_envelope.dat

    uv run generate_angular_momentum.py --model models/scale_CO_core.mod \
        --J_CO 8.597e49 --J_total 8.597e49 --fout data_J_scale_CO_core.dat
"""

import argparse
import re

import numpy as np
from numpy import trapezoid

MSUN_CGS = 1.989e33  # g
G_CGS = 6.67430e-8  # cm^3 g^-1 s^-2


def check_critical_rotation(omega_surface, M_star, R_surface):
    """Report omega_surface / omega_crit at the stellar surface.

    omega_crit is the Keplerian break-up frequency: sqrt(G*M / R^3).
    Warns if the ratio exceeds 0.9 (rotation is super- or near-critical).
    """
    omega_crit = np.sqrt(G_CGS * M_star / R_surface**3)
    ratio = omega_surface / omega_crit
    print(f"  Surface check: omega_surf = {omega_surface:.4e} rad/s, "
          f"omega_crit = {omega_crit:.4e} rad/s")
    print(f"  omega_surf / omega_crit = {ratio:.4f}")
    if ratio >= 1.0:
        print(f"  WARNING: surface is super-critical (ratio >= 1)")
    elif ratio >= 0.9:
        print(f"  WARNING: surface rotation is near critical (ratio >= 0.9)")
    return ratio


def apply_critical_cap(j, r, M_star, q_face, f_crit):
    """Cap j(r) so omega(r) <= f_crit * omega_crit_local(r) at every shell.

    omega_crit_local(r) = sqrt(G * M_enc(r) / r^3), with
    M_enc(r) = M_star * (1 - q_face) — the mass interior to that shell
    (q_face[0] = 0 at surface, increasing inward, so 1 - q_face is the
    enclosed mass fraction).

    Inverts j = (2/3) * omega * r^2 to recover per-cell omega, applies
    the cap, then rebuilds j. Returns the new j and a stats dict.
    """
    omega_cell = j / ((2.0 / 3.0) * r**2)
    M_enc = M_star * (1.0 - q_face)
    omega_crit_local = np.sqrt(G_CGS * M_enc / r**3)
    cap = f_crit * omega_crit_local

    capped_mask = omega_cell > cap
    omega_capped = np.where(capped_mask, cap, omega_cell)
    j_new = (2.0 / 3.0) * omega_capped * r**2

    ratio_before = omega_cell / omega_crit_local
    ratio_after = omega_capped / omega_crit_local
    stats = {
        "n_capped": int(capped_mask.sum()),
        "n_total": len(j),
        "max_ratio_before": float(ratio_before.max()),
        "max_ratio_after": float(ratio_after.max()),
        "first_capped_q": float(q_face[capped_mask][0]) if capped_mask.any() else None,
    }
    return j_new, stats


def load_mesa_model(filepath):
    """Load a MESA .mod file and return key profile quantities."""
    with open(filepath) as f:
        lines = f.readlines()

    metadata = {}
    header_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("!") or stripped == "":
            continue
        if stripped.startswith("lnd"):
            header_line_idx = i
            break
        m = re.match(r"\s*(\S+)\s+(.*)", stripped)
        if m:
            metadata[m.group(1)] = m.group(2).strip()

    if header_line_idx is None:
        raise ValueError("Could not find column header line in model file")

    col_names = lines[header_line_idx].split()

    data_lines = []
    for line in lines[header_line_idx + 1 :]:
        stripped = line.strip()
        if stripped == "":
            continue
        parts = stripped.split()
        if not parts[0].isdigit():
            break
        stripped = stripped.replace("D+", "E+").replace("D-", "E-")
        stripped = re.sub(r"(\d)([-+]\d{2,})", r"\1E\2", stripped)
        parts = stripped.split()
        data_lines.append([float(x) for x in parts[1:]])

    data = np.array(data_lines)
    col_idx = {name: i for i, name in enumerate(col_names)}

    lnR = data[:, col_idx["lnR"]]
    dq = data[:, col_idx["dq"]]

    q_face = np.zeros(len(dq))
    q_face[1:] = np.cumsum(dq[:-1])

    return {
        "radius_cm": np.exp(lnR),
        "dq": dq,
        "q_face": q_face,
        "he4": data[:, col_idx["he4"]],
        "c12": data[:, col_idx["c12"]],
        "metadata": metadata,
    }


def generate_angular_momentum(model, J_CO, J_total, outfile, f_crit=None):
    """Generate angular momentum profile and write to file.

    Args:
        model: dict from load_mesa_model
        J_CO: total angular momentum of CO core (g cm^2 s^-1)
        J_total: total angular momentum of the entire star (g cm^2 s^-1)
        outfile: output file path
        f_crit: if set, cap omega(r) at f_crit * omega_crit_local(r) per shell
            (e.g. 0.5). Solid-body omegas are still computed from J_CO/J_total
            so the core keeps its full ω; the cap only bites where the
            solid-body profile exceeds the local Keplerian limit, mostly in
            the outer envelope. Total J in the output drops accordingly.

    Returns:
        xq, j, interface_q (or None if purely CO core).
    """
    J_He = J_total - J_CO
    M_star = float(model["metadata"]["M/Msun"].replace("D", "E")) * MSUN_CGS
    r = model["radius_cm"]
    dq = model["dq"]
    he4 = model["he4"]
    c12 = model["c12"]
    xq = model["q_face"]

    # Find He/CO interface: where he4 crosses c12 going inward.
    # If no crossing exists, the model is purely CO core.
    diff = he4 - c12
    has_he_envelope = False
    interface_idx = None
    for i in range(len(diff) - 1):
        if diff[i] > 0 and diff[i + 1] <= 0:
            interface_idx = i
            has_he_envelope = True
            break

    interface_q = xq[interface_idx] if has_he_envelope else None

    # For solid body rotation: j(r) = (2/3) * omega * r^2
    # Normalize omega via trapezoidal integration so MESA recovers correct J:
    #   J = M_star * trapz(j, xq)
    r2_profile = (2.0 / 3.0) * r**2
    j = np.empty(len(r))

    if has_he_envelope:
        print(f"  He/CO interface at zone {interface_idx + 1}, q = {interface_q:.4f}")

        he_mask = slice(0, interface_idx + 1)
        co_mask = slice(interface_idx + 1, None)

        I_He = M_star * trapezoid(r2_profile[he_mask], xq[he_mask])
        omega_He = J_He / I_He if J_He > 0 else 0.0

        I_CO = M_star * trapezoid(r2_profile[co_mask], xq[co_mask])
        omega_CO = J_CO / I_CO

        print(f"  omega_He = {omega_He:.4e} rad/s,  omega_CO = {omega_CO:.4e} rad/s")

        j[he_mask] = (2.0 / 3.0) * omega_He * r[he_mask] ** 2
        j[co_mask] = (2.0 / 3.0) * omega_CO * r[co_mask] ** 2
        omega_surface = omega_He
    else:
        print(f"  Purely CO core (no he4=c12 crossing)")
        I_total = M_star * trapezoid(r2_profile, xq)
        omega = J_total / I_total
        print(f"  omega = {omega:.4e} rad/s")
        j[:] = (2.0 / 3.0) * omega * r**2
        omega_surface = omega

    # Apply per-shell critical-omega cap (preserves core, suppresses envelope).
    if f_crit is not None:
        J_pre = M_star * trapezoid(j, xq)
        j, stats = apply_critical_cap(j, r, M_star, xq, f_crit)
        J_post = M_star * trapezoid(j, xq)
        print(f"  Cap (f_crit = {f_crit}): {stats['n_capped']}/{stats['n_total']} "
              f"shells capped")
        if stats["first_capped_q"] is not None:
            print(f"    cap first binds at q = {stats['first_capped_q']:.4f}")
        print(f"    max omega/omega_crit_local: "
              f"{stats['max_ratio_before']:.4f} -> {stats['max_ratio_after']:.4f}")
        print(f"    J: {J_pre:.4e} -> {J_post:.4e} "
              f"({100 * (J_pre - J_post) / J_pre:.2f}% lost to cap)")
        # Surface omega may have been capped — reflect that in the surface check.
        omega_surface = j[0] / ((2.0 / 3.0) * r[0] ** 2)

    # r[0] is the outer face of zone 1 (the surface) — see CLAUDE.md on conventions.
    check_critical_rotation(omega_surface, M_star, r[0])

    # Write data file
    with open(outfile, "w") as f:
        f.write(f"{len(xq)}\n")
        for i in range(len(xq)):
            f.write(f"  {xq[i]:.17e}  {j[i]:.17e}\n")
    print(f"  Wrote {outfile}")

    # Verify
    J_check = M_star * trapezoid(j, xq)
    print(f"  J_total check (trapz): {J_check:.6e} g cm^2/s")

    return xq, j, interface_q


def main():
    parser = argparse.ArgumentParser(
        description="Generate MESA angular momentum profile (solid body rotation)."
    )
    parser.add_argument("--model", required=True, help="Path to MESA .mod file")
    parser.add_argument(
        "--J_CO", required=True, type=float,
        help="Total angular momentum of CO core (g cm^2 s^-1)",
    )
    parser.add_argument(
        "--J_total", required=True, type=float,
        help="Total angular momentum of the entire star (g cm^2 s^-1)",
    )
    parser.add_argument("--fout", required=True, help="Output data file path")
    parser.add_argument(
        "--f_crit", type=float, default=None,
        help="Cap omega per shell at f_crit * sqrt(G*M_enc/r^3) (e.g. 0.5). "
             "Default: no cap.",
    )
    parser.add_argument(
        "--plot", default=None,
        help="Save a plot of j(q) to this file (optional)",
    )
    args = parser.parse_args()

    print(f"Loading {args.model}")
    model = load_mesa_model(args.model)
    print(f"  {len(model['q_face'])} zones, M = {model['metadata']['M/Msun']} Msun")

    print(f"Generating J profile: J_CO = {args.J_CO:.4e}, J_total = {args.J_total:.4e}")
    xq, j, interface_q = generate_angular_momentum(
        model, args.J_CO, args.J_total, args.fout, f_crit=args.f_crit
    )

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(xq, j)
        if interface_q is not None:
            ax.axvline(interface_q, ls="--", color="gray", alpha=0.7, label="He/CO interface")
            ax.legend()
        ax.set_xlabel("q  (0 = surface, 1 = center)")
        ax.set_ylabel(r"Specific angular momentum (cm$^2$/s)")
        ax.set_title(f"Angular Momentum Profile")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"  Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
