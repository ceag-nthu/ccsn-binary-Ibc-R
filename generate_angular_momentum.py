"""Generate MESA angular momentum profile (solid body / flat-j / omega-crit law).

Usage:
    # Solid-body rotation (default)
    uv run generate_angular_momentum.py --model models/adjust_He_envelope.mod \
        --J_CO 8.597e49 --J_total 8.597e49 --fout data_J_adjust_He_envelope.dat

    # Flat-j: omega(r) = omega0 / (1 + (r/A)^2)^p. With --max_omega_ratio,
    # the whole profile is uniformly scaled so max(omega/omega_crit_local) = Y.
    uv run generate_angular_momentum.py --model models/adjust_He_envelope_2.mod \
        --J_CO 1.996e51 --J_total 1.996e51 --profile flat_j --f_core 0.5 --p 2 \
        --max_omega_ratio 0.5 --fout data_J_adjust_He_envelope_2.dat

    # Omega-crit law: omega(r) = Y * sqrt(G * M_enc(r) / r^3) at every shell.
    # Y is solved from the input J (per region). Maximizes J for a given safety
    # factor; no f_core / p / max_omega_ratio knobs needed.
    uv run generate_angular_momentum.py --model models/adjust_He_envelope_2.mod \
        --J_CO 1.996e51 --J_total 1.996e51 --profile omega_crit_law \
        --fout data_J_adjust_He_envelope_2.dat
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


def build_shape(profile, r_region, R_outer, f_scale, p=1.0, M_enc_region=None):
    """Return (shape, A) where shape = omega(r) / omega0 for the given profile.

    profile = "solid_body":     shape = 1 (uniform omega within the region).
    profile = "flat_j":         shape = 1 / (1 + (r/A)^2)^p, A = f_scale * R_outer.
        - p=1: omega ~ 1/r^2 at r >> A, j flattens to (2/3) omega0 A^2.
        - p>1: omega ~ 1/r^(2p), j(r) peaks near r = A/sqrt(p-1) and decreases
          outward — keeps the surface subcritical without a tiny A.
    profile = "omega_crit_law": shape = sqrt(M_enc / r^3). After solve_omega0,
        omega(r) / omega_crit_local(r) = omega0 / sqrt(G_CGS) is *uniform* in
        radius. The constant Y = omega0/sqrt(G_CGS) is reported; if Y >= 1 the
        requested J exceeds what this structure can hold subcritically.
    A is None for solid_body and omega_crit_law.
    """
    if profile == "solid_body":
        return np.ones_like(r_region), None
    if profile == "flat_j":
        A = f_scale * R_outer
        return 1.0 / (1.0 + (r_region / A) ** 2) ** p, A
    if profile == "omega_crit_law":
        if M_enc_region is None:
            raise ValueError("omega_crit_law requires M_enc_region")
        return np.sqrt(M_enc_region / r_region**3), None
    raise ValueError(f"Unknown profile: {profile!r}")


def solve_omega0(shape, r_region, xq_region, M_star, J_target):
    """Solve omega0 such that M_star * trapz((2/3)*omega0*shape*r^2, xq) = J_target.

    Trapezoidal quadrature is used to match how MESA integrates the input
    j(q) profile back to a total J — see CLAUDE.md.
    """
    if J_target <= 0:
        return 0.0
    integral = M_star * trapezoid((2.0 / 3.0) * shape * r_region**2, xq_region)
    return J_target / integral


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


def generate_angular_momentum(model, J_CO, J_total, outfile, profile="solid_body",
                              f_core=0.2, f_he=0.5, p=1.0,
                              max_omega_ratio=None, f_crit=None):
    """Generate angular momentum profile and write to file.

    Args:
        model: dict from load_mesa_model
        J_CO: total angular momentum of CO core (g cm^2 s^-1)
        J_total: total angular momentum of the entire star (g cm^2 s^-1)
        outfile: output file path
        profile: rotation profile shape, applied piecewise per region:
            "solid_body"     — uniform omega within each region (default).
            "flat_j"         — omega(r) = omega0 / (1 + (r/A)^2)^p.
            "omega_crit_law" — omega(r) = Y * sqrt(G * M_enc / r^3); Y is solved
                from the input J. Reports Y per region; warns if Y >= 1.
        f_core: scale for CO region: A_CO = f_core * R_CO_outer. Only used for
            profile="flat_j".
        f_he: scale for He envelope: A_He = f_he * R_surface. Only used for
            profile="flat_j" with a He envelope present.
        p: power in the flat_j shape factor 1/(1+(r/A)^2)^p.
        max_omega_ratio: if set, after building the profile rescale omega(r)
            uniformly so that max(omega/omega_crit_local) equals this value.
            Smooth alternative to f_crit (which clips per shell). J drops
            linearly with the rescale factor. Applied before f_crit.
        f_crit: if set, cap omega(r) at f_crit * omega_crit_local(r) per shell
            after the profile is built. Applied after max_omega_ratio.

    Returns:
        xq, j, interface_q (or None if purely CO core).
    """
    J_He = J_total - J_CO
    M_star = float(model["metadata"]["M/Msun"].replace("D", "E")) * MSUN_CGS
    r = model["radius_cm"]
    he4 = model["he4"]
    c12 = model["c12"]
    xq = model["q_face"]
    # Enclosed mass at each face: q_face = 0 at surface (M_enc = M_star), -> 1 at center.
    M_enc = M_star * (1.0 - xq)

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
    j = np.empty(len(r))

    if has_he_envelope:
        print(f"  He/CO interface at zone {interface_idx + 1}, q = {interface_q:.4f}")

        he_mask = slice(0, interface_idx + 1)
        co_mask = slice(interface_idx + 1, None)

        # zone 0 is the surface (largest r), so r[mask][0] is the outer radius.
        R_He_outer = r[he_mask][0]   # surface
        R_CO_outer = r[co_mask][0]   # top of CO core, just inside interface

        shape_He, A_He = build_shape(profile, r[he_mask], R_He_outer, f_he,
                                      p=p, M_enc_region=M_enc[he_mask])
        shape_CO, A_CO = build_shape(profile, r[co_mask], R_CO_outer, f_core,
                                      p=p, M_enc_region=M_enc[co_mask])

        omega0_He = solve_omega0(shape_He, r[he_mask], xq[he_mask], M_star, J_He)
        omega0_CO = solve_omega0(shape_CO, r[co_mask], xq[co_mask], M_star, J_CO)

        omega_He_arr = omega0_He * shape_He
        omega_CO_arr = omega0_CO * shape_CO
        j[he_mask] = (2.0 / 3.0) * omega_He_arr * r[he_mask] ** 2
        j[co_mask] = (2.0 / 3.0) * omega_CO_arr * r[co_mask] ** 2

        if profile == "flat_j":
            print(f"  flat_j (p={p}): A_He = {A_He:.4e} cm (f_he={f_he}), "
                  f"A_CO = {A_CO:.4e} cm (f_core={f_core})")
            print(f"  omega0_He = {omega0_He:.4e}, omega0_CO = {omega0_CO:.4e} rad/s")
            print(f"  omega(surface) = {omega_He_arr[0]:.4e}, "
                  f"omega(interface He side) = {omega_He_arr[-1]:.4e}, "
                  f"(CO side) = {omega_CO_arr[0]:.4e} rad/s")
        elif profile == "omega_crit_law":
            Y_He = omega0_He / np.sqrt(G_CGS) if omega0_He > 0 else 0.0
            Y_CO = omega0_CO / np.sqrt(G_CGS)
            print(f"  omega_crit_law: Y_He = {Y_He:.4f}, Y_CO = {Y_CO:.4f}  "
                  f"(omega/omega_crit_local is uniform per region)")
            for label, Y_, J_, I_ in [
                ("He", Y_He, J_He, M_star * trapezoid((2.0/3.0) * shape_He * r[he_mask]**2, xq[he_mask])),
                ("CO", Y_CO, J_CO, M_star * trapezoid((2.0/3.0) * shape_CO * r[co_mask]**2, xq[co_mask])),
            ]:
                J_max = np.sqrt(G_CGS) * I_
                print(f"    {label}: J_target = {J_:.3e}, J_max(Y=1) = {J_max:.3e}")
                if Y_ >= 1.0:
                    print(f"    WARNING: Y_{label} >= 1 — requested J exceeds capacity")
        else:
            print(f"  omega_He = {omega0_He:.4e} rad/s, omega_CO = {omega0_CO:.4e} rad/s")

        omega_surface = omega_He_arr[0]
    else:
        if J_CO != J_total:
            print(f"  WARNING: purely CO model, J_He = {J_He:.3e} ignored "
                  f"(using J_total = {J_total:.3e} for whole star)")
        R_outer = r[0]
        shape, A = build_shape(profile, r, R_outer, f_core, p=p, M_enc_region=M_enc)
        omega0 = solve_omega0(shape, r, xq, M_star, J_total)
        omega_arr = omega0 * shape
        j[:] = (2.0 / 3.0) * omega_arr * r**2

        if profile == "flat_j":
            print(f"  Purely CO core, flat_j (p={p}): A = {A:.4e} cm (f_core={f_core})")
            print(f"  omega0 = {omega0:.4e} rad/s, omega(surface) = "
                  f"{omega_arr[0]:.4e} rad/s")
        elif profile == "omega_crit_law":
            Y = omega0 / np.sqrt(G_CGS)
            I_unit = M_star * trapezoid((2.0/3.0) * shape * r**2, xq)
            J_max = np.sqrt(G_CGS) * I_unit
            print(f"  Purely CO core, omega_crit_law: Y = {Y:.4f} "
                  f"(uniform omega/omega_crit_local)")
            print(f"    J_target = {J_total:.3e}, J_max(Y=1) = {J_max:.3e}")
            if Y >= 1.0:
                print(f"    WARNING: Y >= 1 — requested J exceeds physical capacity")
        else:
            print(f"  Purely CO core (no he4=c12 crossing)")
            print(f"  omega = {omega0:.4e} rad/s")

        omega_surface = omega_arr[0]

    # Uniform downscale to enforce max(omega/omega_crit_local) <= max_omega_ratio.
    # Smooth alternative to f_crit: preserves the profile shape, but trades J.
    if max_omega_ratio is not None:
        omega_cell = j / ((2.0 / 3.0) * r**2)
        omega_crit_local = np.sqrt(G_CGS * M_enc / r**3)
        ratio = omega_cell / omega_crit_local
        max_ratio = float(ratio.max())
        if max_ratio > max_omega_ratio:
            scale = max_omega_ratio / max_ratio
            J_pre = M_star * trapezoid(j, xq)
            j = j * scale
            J_post = M_star * trapezoid(j, xq)
            print(f"  max_omega_ratio = {max_omega_ratio}: max ratio "
                  f"{max_ratio:.4f} -> {max_omega_ratio:.4f}, scale = {scale:.4f}")
            print(f"    J: {J_pre:.4e} -> {J_post:.4e} "
                  f"({100 * (1 - scale):.2f}% reduction)")
            omega_surface = j[0] / ((2.0 / 3.0) * r[0] ** 2)
        else:
            print(f"  max_omega_ratio = {max_omega_ratio}: max ratio is "
                  f"{max_ratio:.4f}, no rescale needed")

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
        description="Generate MESA angular momentum profile (solid body or flat-j)."
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
        "--profile",
        choices=["solid_body", "flat_j", "omega_crit_law"],
        default="solid_body",
        help="Rotation profile shape, applied per region. 'solid_body' "
             "(default) uses uniform omega per region; 'flat_j' uses "
             "omega(r) = omega0 / (1 + (r/A)^2)^p; 'omega_crit_law' uses "
             "omega(r) = Y * sqrt(G*M_enc/r^3), so omega/omega_crit_local "
             "is uniform and J is maximized for a given safety factor.",
    )
    parser.add_argument(
        "--f_core", type=float, default=0.2,
        help="For --profile flat_j: A_CO = f_core * R_CO_outer. Smaller -> "
             "more concentrated in the very center, lower surface omega. "
             "Default: 0.2.",
    )
    parser.add_argument(
        "--f_he", type=float, default=0.5,
        help="For --profile flat_j with a He envelope: A_He = f_he * R_surface. "
             "Default: 0.5.",
    )
    parser.add_argument(
        "--p", type=float, default=1.0,
        help="For --profile flat_j: power in shape = 1/(1+(r/A)^2)^p. "
             "p=1 (default) gives j -> const at large r; p=2 makes omega "
             "drop as 1/r^4 so j peaks at r=A and decreases outward — useful "
             "when p=1 leaves the surface too close to critical without "
             "wanting to use a tiny f_core.",
    )
    parser.add_argument(
        "--max_omega_ratio", type=float, default=None,
        help="If set, uniformly rescale the profile so that "
             "max(omega/omega_crit_local) = this value. Smooth alternative to "
             "--f_crit (which clips per shell). J drops linearly with the "
             "rescale factor. Applied before --f_crit. Default: no rescale.",
    )
    parser.add_argument(
        "--f_crit", type=float, default=None,
        help="Cap omega per shell at f_crit * sqrt(G*M_enc/r^3) (e.g. 0.5). "
             "Applied after --max_omega_ratio. Default: no cap.",
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
        model, args.J_CO, args.J_total, args.fout,
        profile=args.profile, f_core=args.f_core, f_he=args.f_he, p=args.p,
        max_omega_ratio=args.max_omega_ratio, f_crit=args.f_crit,
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
