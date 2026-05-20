#!/usr/bin/env python3
"""
Run Orekit angles-only IOD methods on a CSV of spaceborne angular observations.

Supported Orekit IOD classes:
  - IodGauss    angles-only, 3 LOS + 3 observer positions
  - IodLaplace  angles-only, 3 LOS + observer PV at central epoch
  - IodGooding  angles-only, 3 LOS + 3 observer positions + endpoint range guesses

This runner is YAML-configured and writes one result row per requested method. A
method failure is recorded in the output CSV rather than stopping the run.

Expected CSV columns by default:
  EPOCH(JDTDB)
  SC_GEO_X/Y/Z(KM), SC_GEO_VX/Y/Z(KM/S)
  SIN_RA, COS_RA, SIN_DEC
  GEO_X/Y/Z(KM), GEO_VX/Y/Z(KM/S)    # optional, needed for truth error metrics

Usage:
  conda activate orekit-gooding
  python run_orekit_angles_only_iod_from_yaml.py iod_angles_only_config.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing PyYAML. Install with: conda install -c conda-forge pyyaml") from exc

M_PER_KM = 1000.0
KM_PER_M = 1.0 / 1000.0
SEC_PER_DAY = 86400.0


@dataclass
class Observation:
    row_index: int
    jd_tdb: float
    date: Any
    obs_r_m: np.ndarray
    obs_v_mps: np.ndarray
    los: np.ndarray
    truth_r_m: Optional[np.ndarray] = None
    truth_v_mps: Optional[np.ndarray] = None


def deep_get(d: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return np.asarray(v, dtype=float) / n


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def los_from_sin_cos(sin_ra: float, cos_ra: float, sin_dec: float) -> np.ndarray:
    sin_ra = float(sin_ra)
    cos_ra = float(cos_ra)
    sin_dec = float(sin_dec)
    # Normalize RA sine/cosine in case they are not exactly unit length.
    ra_norm = math.hypot(sin_ra, cos_ra)
    if ra_norm == 0.0:
        raise ValueError("SIN_RA and COS_RA are both zero")
    sin_ra /= ra_norm
    cos_ra /= ra_norm
    sin_dec = max(-1.0, min(1.0, sin_dec))
    cos_dec = math.sqrt(max(0.0, 1.0 - sin_dec * sin_dec))
    return unit(np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec], dtype=float))


def rel_angles_rates(rel_r: np.ndarray, rel_v: np.ndarray) -> Dict[str, float]:
    x, y, z = [float(q) for q in rel_r]
    vx, vy, vz = [float(q) for q in rel_v]
    rho = math.sqrt(x*x + y*y + z*z)
    xy2 = x*x + y*y
    xy = math.sqrt(max(0.0, xy2))
    if rho <= 0:
        raise ValueError("zero relative range")
    ra = math.atan2(y, x)
    dec = math.asin(max(-1.0, min(1.0, z / rho)))
    rho_dot = (x*vx + y*vy + z*vz) / rho
    if xy2 > 0.0:
        ra_dot = (x*vy - y*vx) / xy2
    else:
        ra_dot = float("nan")
    # dec = atan2(z, sqrt(x^2+y^2)) is numerically stable for rates
    # ddec = (xy*vz - z*d(xy)/dt) / rho^2, d(xy)/dt=(x vx + y vy)/xy
    if xy > 0.0:
        xy_dot = (x*vx + y*vy) / xy
        dec_dot = (xy * vz - z * xy_dot) / (rho * rho)
    else:
        dec_dot = float("nan")
    return {
        "ra_rad": ra,
        "dec_rad": dec,
        "rho_m": rho,
        "ra_rate_rad_s": ra_dot,
        "dec_rate_rad_s": dec_dot,
        "rho_rate_m_s": rho_dot,
    }


def vector3d_to_np(v: Any) -> np.ndarray:
    return np.array([float(v.getX()), float(v.getY()), float(v.getZ())], dtype=float)


def np_to_vector3d(v: np.ndarray) -> Any:
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    return Vector3D(float(v[0]), float(v[1]), float(v[2]))


def setup_orekit(orekit_data_path: Optional[str]) -> None:
    import orekit
    orekit.initVM()

    if orekit_data_path:
        from java.io import File
        from org.orekit.data import DataContext, DirectoryCrawler
        p = Path(orekit_data_path).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"orekit_data_path does not exist or is not a directory: {p}")
        manager = DataContext.getDefault().getDataProvidersManager()
        manager.addProvider(DirectoryCrawler(File(str(p))))
    else:
        # Try the classic helper if available and an orekit-data folder is in cwd.
        try:
            from orekit.pyhelpers import setup_orekit_curdir
            setup_orekit_curdir()
        except Exception:
            pass


def get_frame(frame_name: str) -> Any:
    from org.orekit.frames import FramesFactory
    name = str(frame_name).upper()
    if name in ("EME2000", "EME", "J2000"):
        return FramesFactory.getEME2000()
    if name in ("GCRF",):
        return FramesFactory.getGCRF()
    raise ValueError(f"Unsupported frame_name={frame_name!r}. Add it in get_frame().")


def absolute_date_from_jd_tdb(jd: float) -> Any:
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    tdb = TimeScalesFactory.getTDB()
    jd0 = math.floor(float(jd))
    seconds = (float(jd) - jd0) * SEC_PER_DAY
    # Orekit supports this static factory in modern versions.
    try:
        return AbsoluteDate.createJDDate(int(jd0), float(seconds), tdb)
    except Exception:
        # Fallback: shift from J2000 epoch in TDB-compatible seconds.
        return AbsoluteDate.J2000_EPOCH.shiftedBy((float(jd) - 2451545.0) * SEC_PER_DAY)


def parse_rows(rows_cfg: Any, n_rows: int) -> List[int]:
    if rows_cfg is None or rows_cfg == "first_middle_last":
        return [0, n_rows // 2, n_rows - 1]
    if isinstance(rows_cfg, str):
        s = rows_cfg.strip().lower()
        if s in ("first middle last", "first,middle,last", "first_middle_last"):
            return [0, n_rows // 2, n_rows - 1]
        raise ValueError(f"Unsupported rows string: {rows_cfg}")
    if isinstance(rows_cfg, (list, tuple)):
        out = []
        for r in rows_cfg:
            if isinstance(r, str):
                rr = r.lower()
                if rr == "first":
                    out.append(0)
                elif rr == "middle":
                    out.append(n_rows // 2)
                elif rr == "last":
                    out.append(n_rows - 1)
                else:
                    out.append(int(r))
            else:
                out.append(int(r))
        if len(out) != 3:
            raise ValueError("rows must have exactly 3 entries for angles-only IOD methods")
        return out
    raise ValueError(f"Unsupported rows config: {rows_cfg!r}")


def load_observations(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Observation]]:
    csv_path = deep_get(cfg, ["input", "csv_path"])
    if not csv_path:
        raise ValueError("config input.csv_path is required")
    df = pd.read_csv(Path(csv_path).expanduser())
    col = deep_get(cfg, ["columns"], {})
    epoch_col = col.get("epoch_jd_tdb", "EPOCH(JDTDB)")
    obs_pos_cols = col.get("observer_position_km", ["SC_GEO_X(KM)", "SC_GEO_Y(KM)", "SC_GEO_Z(KM)"])
    obs_vel_cols = col.get("observer_velocity_km_s", ["SC_GEO_VX(KM/S)", "SC_GEO_VY(KM/S)", "SC_GEO_VZ(KM/S)"])
    los_cols = col.get("los_sin_cos", ["SIN_RA", "COS_RA", "SIN_DEC"])
    truth_pos_cols = col.get("truth_position_km", ["GEO_X(KM)", "GEO_Y(KM)", "GEO_Z(KM)"])
    truth_vel_cols = col.get("truth_velocity_km_s", ["GEO_VX(KM/S)", "GEO_VY(KM/S)", "GEO_VZ(KM/S)"])

    require_cols(df, [epoch_col] + obs_pos_cols + obs_vel_cols + los_cols)
    has_truth = all(c in df.columns for c in truth_pos_cols + truth_vel_cols)

    obs_list: List[Observation] = []
    for idx, row in df.iterrows():
        jd = float(row[epoch_col])
        obs_r_m = np.array([float(row[c]) for c in obs_pos_cols], dtype=float) * M_PER_KM
        obs_v_mps = np.array([float(row[c]) for c in obs_vel_cols], dtype=float) * M_PER_KM
        los = los_from_sin_cos(float(row[los_cols[0]]), float(row[los_cols[1]]), float(row[los_cols[2]]))
        if has_truth:
            tr = np.array([float(row[c]) for c in truth_pos_cols], dtype=float) * M_PER_KM
            tv = np.array([float(row[c]) for c in truth_vel_cols], dtype=float) * M_PER_KM
        else:
            tr = None
            tv = None
        obs_list.append(Observation(int(idx), jd, absolute_date_from_jd_tdb(jd), obs_r_m, obs_v_mps, los, tr, tv))
    return df, obs_list


def truth_range_m(o: Observation) -> float:
    if o.truth_r_m is None:
        raise ValueError("truth position is unavailable; cannot use truth-derived range")
    return norm(o.truth_r_m - o.obs_r_m)


def orbit_to_state_at_date(orbit: Any, frame: Any, target_date: Any) -> Tuple[np.ndarray, np.ndarray]:
    try:
        dt = float(target_date.durationFrom(orbit.getDate()))
        if abs(dt) > 1e-12:
            orbit_eval = orbit.shiftedBy(dt)
        else:
            orbit_eval = orbit
    except Exception:
        orbit_eval = orbit
    try:
        pv = orbit_eval.getPVCoordinates(frame)
    except Exception:
        pv = orbit_eval.getPVCoordinates()
    return vector3d_to_np(pv.getPosition()), vector3d_to_np(pv.getVelocity())


def selected_obs(all_obs: List[Observation], rows: List[int]) -> List[Observation]:
    n = len(all_obs)
    out = []
    for r in rows:
        rr = r if r >= 0 else n + r
        if rr < 0 or rr >= n:
            raise IndexError(f"row index {r} resolved to {rr}, outside [0,{n-1}]")
        out.append(all_obs[rr])
    return out


def method_list(method_cfg: Any) -> List[str]:
    all_methods = ["gauss", "laplace", "gooding"]
    if method_cfg is None:
        return all_methods
    if isinstance(method_cfg, str):
        if method_cfg.lower() == "all":
            return all_methods
        return [method_cfg.lower()]
    methods = [str(m).lower() for m in method_cfg]
    if "all" in methods:
        return all_methods
    unknown = [m for m in methods if m not in all_methods]
    if unknown:
        raise ValueError(
            f"Unsupported angles-only method(s): {unknown}. "
            f"Allowed values are: {all_methods} or 'all'."
        )
    return methods


def call_iod_method(method: str, cfg: Dict[str, Any], frame: Any, obs3: List[Observation]) -> Tuple[Any, Dict[str, Any]]:
    from org.orekit.estimation.iod import IodGauss, IodGooding, IodLaplace
    from org.orekit.utils import PVCoordinates

    mu = float(deep_get(cfg, ["orekit", "mu_m3_s2"], 3.986004418e14))
    method = method.lower()
    meta: Dict[str, Any] = {
        "method_category": "angles_only",
        "uses_truth_ranges_or_positions": False,
    }

    if len(obs3) != 3:
        raise ValueError(f"{method} requires exactly 3 selected rows")

    if method == "gauss":
        iod = IodGauss(mu)
        return iod.estimate(
            frame,
            np_to_vector3d(obs3[0].obs_r_m), obs3[0].date, np_to_vector3d(obs3[0].los),
            np_to_vector3d(obs3[1].obs_r_m), obs3[1].date, np_to_vector3d(obs3[1].los),
            np_to_vector3d(obs3[2].obs_r_m), obs3[2].date, np_to_vector3d(obs3[2].los),
        ), meta

    if method == "laplace":
        iod = IodLaplace(mu)
        mid = obs3[1]
        obs_pv = PVCoordinates(np_to_vector3d(mid.obs_r_m), np_to_vector3d(mid.obs_v_mps))
        meta["note"] = "Orekit Laplace LOS overload uses observer PV at the central observation."
        return iod.estimate(
            frame,
            obs_pv,
            obs3[0].date, np_to_vector3d(obs3[0].los),
            obs3[1].date, np_to_vector3d(obs3[1].los),
            obs3[2].date, np_to_vector3d(obs3[2].los),
        ), meta

    if method == "gooding":
        iod = IodGooding(mu)
        gcfg = deep_get(cfg, ["gooding"], {}) or {}
        range_init = str(gcfg.get("range_init", "truth")).lower()

        if range_init == "truth":
            rho1_m = truth_range_m(obs3[0])
            rho3_m = truth_range_m(obs3[2])
            meta["uses_truth_ranges_or_positions"] = True
            meta["note"] = (
                "Gooding endpoint ranges initialized from truth; useful for validation/failure-rate "
                "testing, but not a blind IOD result."
            )
        elif range_init == "manual":
            rho1_m = float(gcfg["rho1_km"]) * M_PER_KM
            rho3_m = float(gcfg["rho3_km"]) * M_PER_KM
        else:
            raise ValueError("gooding.range_init must be 'truth' or 'manual'")

        meta["rho1_init_km"] = rho1_m * KM_PER_M
        meta["rho3_init_km"] = rho3_m * KM_PER_M
        n_rev = int(gcfg.get("n_rev", 0))
        dirs = gcfg.get("directions_to_try", [True, False])
        if isinstance(dirs, bool):
            dirs = [dirs]

        errors = []
        for direction in dirs:
            try:
                orbit = iod.estimate(
                    frame,
                    np_to_vector3d(obs3[0].obs_r_m),
                    np_to_vector3d(obs3[1].obs_r_m),
                    np_to_vector3d(obs3[2].obs_r_m),
                    np_to_vector3d(obs3[0].los), obs3[0].date,
                    np_to_vector3d(obs3[1].los), obs3[1].date,
                    np_to_vector3d(obs3[2].los), obs3[2].date,
                    float(rho1_m), float(rho3_m), int(n_rev), bool(direction),
                )
                meta["gooding_direction"] = bool(direction)
                meta["n_rev"] = n_rev
                if errors:
                    meta["previous_direction_errors"] = " | ".join(errors)
                return orbit, meta
            except Exception as exc:
                msg = str(exc).splitlines()[0] if str(exc) else repr(exc)
                errors.append(f"direction={direction}: {type(exc).__name__}: {msg}")

        raise RuntimeError("Gooding failed for all configured directions: " + " | ".join(errors))

    raise ValueError(f"Unknown angles-only method: {method}")

def diagnostic_geometry(obs3: List[Observation]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if len(obs3) >= 2:
        out["dt_first_last_s"] = (obs3[-1].jd_tdb - obs3[0].jd_tdb) * SEC_PER_DAY
        out["los_angle_first_last_deg"] = math.degrees(math.acos(float(np.clip(np.dot(obs3[0].los, obs3[-1].los), -1, 1))))
    if len(obs3) == 3:
        out["dt12_s"] = (obs3[1].jd_tdb - obs3[0].jd_tdb) * SEC_PER_DAY
        out["dt23_s"] = (obs3[2].jd_tdb - obs3[1].jd_tdb) * SEC_PER_DAY
        out["los_angle_12_deg"] = math.degrees(math.acos(float(np.clip(np.dot(obs3[0].los, obs3[1].los), -1, 1))))
        out["los_angle_23_deg"] = math.degrees(math.acos(float(np.clip(np.dot(obs3[1].los, obs3[2].los), -1, 1))))
    return out


def compute_result_row(method: str, status: str, obs_sel: List[Observation], eval_obs: Observation, orbit: Optional[Any], frame: Any, meta: Dict[str, Any], error: Optional[BaseException] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": method,
        "status": status,
        "selected_rows": ";".join(str(o.row_index) for o in obs_sel),
        "eval_row": eval_obs.row_index,
        "eval_epoch_jdtdb": eval_obs.jd_tdb,
    }
    row.update(diagnostic_geometry(obs_sel))
    row.update(meta or {})

    if error is not None:
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error).replace("\n", " | ")[:4000]
        return row

    if orbit is None:
        row["status"] = "null_orbit"
        row["error_message"] = "Orekit returned null/no orbit"
        return row

    r_est_m, v_est_mps = orbit_to_state_at_date(orbit, frame, eval_obs.date)
    row.update({
        "est_x_km": r_est_m[0] * KM_PER_M,
        "est_y_km": r_est_m[1] * KM_PER_M,
        "est_z_km": r_est_m[2] * KM_PER_M,
        "est_vx_km_s": v_est_mps[0] * KM_PER_M,
        "est_vy_km_s": v_est_mps[1] * KM_PER_M,
        "est_vz_km_s": v_est_mps[2] * KM_PER_M,
    })

    if eval_obs.truth_r_m is not None and eval_obs.truth_v_mps is not None:
        dr = r_est_m - eval_obs.truth_r_m
        dv = v_est_mps - eval_obs.truth_v_mps
        row.update({
            "pos_err_km": norm(dr) * KM_PER_M,
            "vel_err_km_s": norm(dv) * KM_PER_M,
            "dx_km": dr[0] * KM_PER_M,
            "dy_km": dr[1] * KM_PER_M,
            "dz_km": dr[2] * KM_PER_M,
            "dvx_km_s": dv[0] * KM_PER_M,
            "dvy_km_s": dv[1] * KM_PER_M,
            "dvz_km_s": dv[2] * KM_PER_M,
        })

        est_rel_r = r_est_m - eval_obs.obs_r_m
        est_rel_v = v_est_mps - eval_obs.obs_v_mps
        tru_rel_r = eval_obs.truth_r_m - eval_obs.obs_r_m
        tru_rel_v = eval_obs.truth_v_mps - eval_obs.obs_v_mps
        est_ar = rel_angles_rates(est_rel_r, est_rel_v)
        tru_ar = rel_angles_rates(tru_rel_r, tru_rel_v)
        ra_err = wrap_to_pi(est_ar["ra_rad"] - tru_ar["ra_rad"])
        dec_err = est_ar["dec_rad"] - tru_ar["dec_rad"]
        rho_err = est_ar["rho_m"] - tru_ar["rho_m"]
        ra_rate_err = est_ar["ra_rate_rad_s"] - tru_ar["ra_rate_rad_s"]
        dec_rate_err = est_ar["dec_rate_rad_s"] - tru_ar["dec_rate_rad_s"]
        rho_rate_err = est_ar["rho_rate_m_s"] - tru_ar["rho_rate_m_s"]
        los_angle_err = math.acos(float(np.clip(np.dot(unit(est_rel_r), unit(tru_rel_r)), -1.0, 1.0)))
        row.update({
            "ra_est_rad": est_ar["ra_rad"],
            "dec_est_rad": est_ar["dec_rad"],
            "rho_est_km": est_ar["rho_m"] * KM_PER_M,
            "ra_rate_est_rad_s": est_ar["ra_rate_rad_s"],
            "dec_rate_est_rad_s": est_ar["dec_rate_rad_s"],
            "rho_rate_est_km_s": est_ar["rho_rate_m_s"] * KM_PER_M,
            "ra_truth_rad": tru_ar["ra_rad"],
            "dec_truth_rad": tru_ar["dec_rad"],
            "rho_truth_km": tru_ar["rho_m"] * KM_PER_M,
            "ra_rate_truth_rad_s": tru_ar["ra_rate_rad_s"],
            "dec_rate_truth_rad_s": tru_ar["dec_rate_rad_s"],
            "rho_rate_truth_km_s": tru_ar["rho_rate_m_s"] * KM_PER_M,
            "ra_err_rad": ra_err,
            "ra_err_arcsec": math.degrees(ra_err) * 3600.0,
            "dec_err_rad": dec_err,
            "dec_err_arcsec": math.degrees(dec_err) * 3600.0,
            "rho_err_km": rho_err * KM_PER_M,
            "ra_rate_err_rad_s": ra_rate_err,
            "ra_rate_err_arcsec_s": math.degrees(ra_rate_err) * 3600.0,
            "dec_rate_err_rad_s": dec_rate_err,
            "dec_rate_err_arcsec_s": math.degrees(dec_rate_err) * 3600.0,
            "rho_rate_err_km_s": rho_rate_err * KM_PER_M,
            "los_angle_err_rad": los_angle_err,
            "los_angle_err_arcsec": math.degrees(los_angle_err) * 3600.0,
        })
    return row


def run(cfg: Dict[str, Any]) -> pd.DataFrame:
    setup_orekit(deep_get(cfg, ["orekit", "data_path"], None))
    frame = get_frame(deep_get(cfg, ["orekit", "frame"], "EME2000"))
    df, all_obs = load_observations(cfg)
    rows = parse_rows(deep_get(cfg, ["observations", "rows"], "first_middle_last"), len(all_obs))
    obs_sel = selected_obs(all_obs, rows)
    eval_choice = deep_get(cfg, ["evaluation", "epoch"], "central_selected")
    if str(eval_choice).lower() in ("central_selected", "middle", "central"):
        eval_obs = obs_sel[len(obs_sel)//2]
    else:
        eval_obs = all_obs[int(eval_choice)]

    methods = method_list(deep_get(cfg, ["methods"], "all"))
    results: List[Dict[str, Any]] = []
    verbose = bool(deep_get(cfg, ["output", "verbose"], True))
    include_traceback = bool(deep_get(cfg, ["output", "include_traceback"], False))

    for method in methods:
        if verbose:
            print(f"Running {method} on rows {[o.row_index for o in obs_sel]} ...", flush=True)
        try:
            orbit, meta = call_iod_method(method, cfg, frame, obs_sel)
            status = "success" if orbit is not None else "null_orbit"
            results.append(compute_result_row(method, status, obs_sel, eval_obs, orbit, frame, meta, cfg=cfg))
        except BaseException as exc:
            if verbose:
                print(f"  {method} failed: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else repr(exc)}", flush=True)
            meta = {}
            err_row = compute_result_row(method, "failed", obs_sel, eval_obs, None, frame, meta, error=exc, cfg=cfg)
            if include_traceback:
                err_row["traceback"] = traceback.format_exc()
            results.append(err_row)

    out_df = pd.DataFrame(results)
    out_csv = deep_get(cfg, ["output", "results_csv"], "iod_results.csv")
    out_path = Path(out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    if verbose:
        print(f"Wrote {out_path}")
    return out_df


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run Orekit angles-only IOD methods from a YAML config.")
    ap.add_argument("config", help="YAML configuration file")
    args = ap.parse_args(argv)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
