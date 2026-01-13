from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader


# =========================
# CONFIGURACIÓN POR DEFECTO
# =========================
DEFAULT_DIR = Path(r"C:\Users\david\Downloads\material")
DEFAULT_BAG = DEFAULT_DIR / "mir_basics_20251210_114529.bag"

# Parámetros EKF (puedes afinarlos)
SIGMA_V = 0.10         # m/s  (ruido del modelo de velocidad lineal)
SIGMA_W = 0.20         # rad/s (ruido del modelo de velocidad angular)
SIGMA_YAW_MEAS = 0.15  # rad  (ruido medida yaw IMU)

# Opción: usar AMCL como "medida" adicional (si te lo permiten)
USE_AMCL_UPDATE = False
SIGMA_POS_MEAS = 0.25  # m


# =========================
# UTILIDADES
# =========================
def wrap_to_pi(a: float) -> float:
    """Normaliza un ángulo a [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_to_yaw(q) -> float:
    """Quaternion -> yaw (rotación Z)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny, cosy))


def interp(t_ref, t_src, y_src):
    """Interpolación lineal para alinear señales por tiempo."""
    t_ref = np.asarray(t_ref, dtype=float)
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    if len(t_src) < 2:
        return np.full_like(t_ref, np.nan, dtype=float)
    return np.interp(t_ref, t_src, y_src)


# =========================
# LECTURA ROSBAG
# =========================
def list_topics(bag_path: Path):
    with AnyReader([bag_path]) as reader:
        return sorted(reader.topics.keys())


def read_pose_topic(bag_path: Path, topic: str) -> pd.DataFrame:
    """
    Lee topics tipo pose:
      msg.pose.pose.position.{x,y}
      msg.pose.pose.orientation.{x,y,z,w}
    Devuelve DataFrame: t, x, y, yaw
    """
    t, x, y, yaw = [], [], [], []
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return pd.DataFrame(columns=["t", "x", "y", "yaw"])

        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            try:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
            except Exception:
                continue

            t.append(ts * 1e-9)
            x.append(float(p.x))
            y.append(float(p.y))
            yaw.append(quat_to_yaw(q))

    df = pd.DataFrame({"t": t, "x": x, "y": y, "yaw": np.unwrap(np.array(yaw, dtype=float))})
    return df.sort_values("t").reset_index(drop=True)


def read_twist_from_odom(bag_path: Path, topic: str = "/odom") -> pd.DataFrame:
    """
    Lee velocidades estimadas (mejor que cmd_vel porque es "lo que pasó"):
      msg.twist.twist.linear.x
      msg.twist.twist.angular.z
    Devuelve DataFrame: t, v, w
    """
    t, v, w = [], [], []
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return pd.DataFrame(columns=["t", "v", "w"])

        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            try:
                v_lin = msg.twist.twist.linear.x
                w_ang = msg.twist.twist.angular.z
            except Exception:
                continue

            t.append(ts * 1e-9)
            v.append(float(v_lin))
            w.append(float(w_ang))

    df = pd.DataFrame({"t": t, "v": v, "w": w})
    return df.sort_values("t").reset_index(drop=True)


def read_imu_yaw_or_wz(bag_path: Path) -> pd.DataFrame:
    """
    IMU:
      - yaw_meas: si orientation quaternion es válido (no todo ceros)
      - wz: angular_velocity.z
    Devuelve DataFrame: t, yaw_meas, wz
    """
    t, yaw_meas, wz = [], [], []
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == "/imu_data"]
        if not conns:
            return pd.DataFrame(columns=["t", "yaw_meas", "wz"])

        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t.append(ts * 1e-9)

            # wz
            try:
                wz.append(float(msg.angular_velocity.z))
            except Exception:
                wz.append(np.nan)

            # yaw (si orientación válida)
            try:
                q = msg.orientation
                if (q.x, q.y, q.z, q.w) != (0.0, 0.0, 0.0, 0.0):
                    yaw_meas.append(quat_to_yaw(q))
                else:
                    yaw_meas.append(np.nan)
            except Exception:
                yaw_meas.append(np.nan)

    df = pd.DataFrame({"t": t, "yaw_meas": yaw_meas, "wz": wz})
    return df.sort_values("t").reset_index(drop=True)


# =========================
# EKF 2D: estado = [x, y, yaw]
# =========================
def ekf_localization(
    vw: pd.DataFrame,
    imu: pd.DataFrame,
    x0: float, y0: float, yaw0: float,
    sigma_v: float, sigma_w: float, sigma_yaw_meas: float,
    amcl: pd.DataFrame | None = None,
    sigma_pos_meas: float = 0.25,
    use_amcl: bool = False,
) -> pd.DataFrame:
    """
    Predicción: modelo unicycle con v,w
    Corrección: yaw IMU si existe
    Corrección opcional: AMCL (x,y,yaw)
    """
    t = vw["t"].to_numpy(dtype=float)
    v = vw["v"].to_numpy(dtype=float)
    w = vw["w"].to_numpy(dtype=float)

    # Alinear yaw IMU a tiempos de t
    t_imu = imu["t"].to_numpy(dtype=float)
    yaw_raw = imu["yaw_meas"].to_numpy(dtype=float)
    has_yaw = np.isfinite(yaw_raw).sum() >= 2

    if has_yaw:
        mask = np.isfinite(yaw_raw)
        yaw_meas = interp(t, t_imu[mask], np.unwrap(yaw_raw[mask]))
    else:
        yaw_meas = np.full_like(t, np.nan, dtype=float)

    # Alinear AMCL (opcional)
    if use_amcl and amcl is not None and len(amcl) >= 2:
        t_amcl = amcl["t"].to_numpy(dtype=float)
        x_amcl = amcl["x"].to_numpy(dtype=float)
        y_amcl = amcl["y"].to_numpy(dtype=float)
        yaw_amcl = np.unwrap(amcl["yaw"].to_numpy(dtype=float))

        x_amcl_i = interp(t, t_amcl, x_amcl)
        y_amcl_i = interp(t, t_amcl, y_amcl)
        yaw_amcl_i = interp(t, t_amcl, yaw_amcl)

        R_amcl = np.diag([sigma_pos_meas**2, sigma_pos_meas**2, (np.deg2rad(10))**2])
    else:
        x_amcl_i = y_amcl_i = yaw_amcl_i = None
        R_amcl = None

    # Estado e incertidumbre
    xk = np.array([x0, y0, yaw0], dtype=float)
    P = np.diag([1.0, 1.0, (np.deg2rad(30))**2])  # incertidumbre inicial

    R_yaw = np.array([[sigma_yaw_meas**2]], dtype=float)

    xs = np.zeros(len(t), dtype=float)
    ys = np.zeros(len(t), dtype=float)
    yaws = np.zeros(len(t), dtype=float)
    xs[0], ys[0], yaws[0] = xk

    for k in range(1, len(t)):
        dt = float(t[k] - t[k - 1])
        if dt <= 0.0:
            dt = 0.0
        if dt > 0.2:
            dt = 0.2  # estabilidad

        vk = float(v[k - 1])
        wk = float(w[k - 1])

        # -------- PREDICCIÓN --------
        yaw_prev = float(xk[2])
        x_pred = float(xk[0] + vk * np.cos(yaw_prev) * dt)
        y_pred = float(xk[1] + vk * np.sin(yaw_prev) * dt)
        yaw_pred = wrap_to_pi(yaw_prev + wk * dt)

        xk = np.array([x_pred, y_pred, yaw_pred], dtype=float)

        # Jacobiano F
        F = np.array([
            [1.0, 0.0, -vk * np.sin(yaw_prev) * dt],
            [0.0, 1.0,  vk * np.cos(yaw_prev) * dt],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        # Ruido de proceso Q
        Q = np.diag([
            (sigma_v * dt) ** 2,
            (sigma_v * dt) ** 2,
            (sigma_w * dt) ** 2,
        ])

        P = F @ P @ F.T + Q

        # -------- UPDATE yaw IMU --------
        if np.isfinite(yaw_meas[k]):
            z = np.array([wrap_to_pi(float(yaw_meas[k]))], dtype=float)
            h = np.array([float(xk[2])], dtype=float)
            H = np.array([[0.0, 0.0, 1.0]], dtype=float)

            # INNOVACIÓN SIN WARNING:
            innov = (z - h).item()   # <-- aquí está la corrección clave
            y_innov = wrap_to_pi(float(innov))

            S = H @ P @ H.T + R_yaw
            K = P @ H.T @ np.linalg.inv(S)

            xk = xk + (K.flatten() * y_innov)
            xk[2] = wrap_to_pi(float(xk[2]))

            P = (np.eye(3) - K @ H) @ P

        # -------- UPDATE AMCL (opcional) --------
        if use_amcl and x_amcl_i is not None:
            if np.isfinite(x_amcl_i[k]) and np.isfinite(y_amcl_i[k]) and np.isfinite(yaw_amcl_i[k]):
                z = np.array([float(x_amcl_i[k]), float(y_amcl_i[k]), wrap_to_pi(float(yaw_amcl_i[k]))], dtype=float)
                h = np.array([float(xk[0]), float(xk[1]), float(xk[2])], dtype=float)
                H = np.eye(3, dtype=float)

                innov = z - h
                innov[2] = wrap_to_pi(float(innov[2]))

                S = H @ P @ H.T + R_amcl
                K = P @ H.T @ np.linalg.inv(S)

                xk = xk + K @ innov
                xk[2] = wrap_to_pi(float(xk[2]))

                P = (np.eye(3) - K @ H) @ P

        xs[k], ys[k], yaws[k] = xk

    return pd.DataFrame({"t": t, "x": xs, "y": ys, "yaw": np.unwrap(yaws)})


# =========================
# MÉTRICAS
# =========================
def ate_rmse(gt: pd.DataFrame, est: pd.DataFrame) -> float:
    """RMSE de posición (m) comparando con GT."""
    t_gt = gt["t"].to_numpy(dtype=float)
    x_est = interp(t_gt, est["t"].to_numpy(dtype=float), est["x"].to_numpy(dtype=float))
    y_est = interp(t_gt, est["t"].to_numpy(dtype=float), est["y"].to_numpy(dtype=float))
    e = np.sqrt((x_est - gt["x"].to_numpy(dtype=float))**2 + (y_est - gt["y"].to_numpy(dtype=float))**2)
    return float(np.sqrt(np.nanmean(e**2)))


def yaw_rmse(gt: pd.DataFrame, est: pd.DataFrame) -> float:
    """RMSE de yaw (rad) comparando con GT."""
    t_gt = gt["t"].to_numpy(dtype=float)
    yaw_est = interp(t_gt, est["t"].to_numpy(dtype=float), est["yaw"].to_numpy(dtype=float))
    yaw_gt = gt["yaw"].to_numpy(dtype=float)
    err = np.array([wrap_to_pi(a - b) for a, b in zip(yaw_est, yaw_gt)], dtype=float)
    return float(np.sqrt(np.nanmean(err**2)))


# =========================
# MAIN
# =========================
def run(bag_path: Path):
    print("Usando rosbag:")
    print(bag_path)

    if not bag_path.exists():
        raise FileNotFoundError(f"No existe el bag en: {bag_path}")

    print("\nTopics disponibles:")
    for tpc in list_topics(bag_path):
        print(" ", tpc)

    # Referencias
    gt = read_pose_topic(bag_path, "/base_pose_ground_truth")
    odom_pose = read_pose_topic(bag_path, "/odom")
    odom_f_pose = read_pose_topic(bag_path, "/odometry/filtered")
    amcl_pose = read_pose_topic(bag_path, "/amcl_pose")

    # Entradas del modelo
    vw = read_twist_from_odom(bag_path, "/odom")      # v,w "reales"
    imu = read_imu_yaw_or_wz(bag_path)                # yaw (si existe) + wz

    print("\nIMU resumen:")
    print(" yaw válidos:", int(np.isfinite(imu['yaw_meas']).sum()), "de", len(imu))
    print(" wz  válidos:", int(np.isfinite(imu['wz']).sum()), "de", len(imu))

    # Estado inicial
    if len(gt) > 0:
        x0, y0, yaw0 = gt.loc[0, ["x", "y", "yaw"]]
    elif len(odom_pose) > 0:
        x0, y0, yaw0 = odom_pose.loc[0, ["x", "y", "yaw"]]
    else:
        x0 = y0 = yaw0 = 0.0

    ekf_est = ekf_localization(
        vw=vw, imu=imu,
        x0=float(x0), y0=float(y0), yaw0=float(yaw0),
        sigma_v=SIGMA_V, sigma_w=SIGMA_W, sigma_yaw_meas=SIGMA_YAW_MEAS,
        amcl=amcl_pose, sigma_pos_meas=SIGMA_POS_MEAS, use_amcl=USE_AMCL_UPDATE
    )

    # Métricas
    if len(gt) > 0:
        print("\nMÉTRICAS (vs GT):")
        if len(odom_pose) > 0:
            print(" ATE RMSE Odom pose (m):       ", ate_rmse(gt, odom_pose))
            print(" Yaw RMSE Odom pose (deg):     ", np.degrees(yaw_rmse(gt, odom_pose)))
        if len(odom_f_pose) > 0:
            print(" ATE RMSE Odom filtered (m):   ", ate_rmse(gt, odom_f_pose))
            print(" Yaw RMSE Odom filtered (deg): ", np.degrees(yaw_rmse(gt, odom_f_pose)))

        print(" ATE RMSE EKF (m):             ", ate_rmse(gt, ekf_est))
        print(" Yaw RMSE EKF (deg):           ", np.degrees(yaw_rmse(gt, ekf_est)))
        print("\nParámetros de 'error' del modelo (sigmas):")
        print(" SIGMA_V:", SIGMA_V, "m/s  | SIGMA_W:", SIGMA_W, "rad/s  | SIGMA_YAW_MEAS:", SIGMA_YAW_MEAS, "rad")
    else:
        print("\nNo hay GT (/base_pose_ground_truth). No se puede calcular RMSE.")

    # Plot
    plt.figure()
    if len(gt) > 0:
        plt.plot(gt.x, gt.y, label="GT")
    if len(odom_pose) > 0:
        plt.plot(odom_pose.x, odom_pose.y, label="Odom pose")
    if len(odom_f_pose) > 0:
        plt.plot(odom_f_pose.x, odom_f_pose.y, label="Odom filtered pose")
    if len(amcl_pose) > 0:
        plt.plot(amcl_pose.x, amcl_pose.y, label="AMCL pose")
    plt.plot(ekf_est.x, ekf_est.y, label=f"EKF (AMCL update={USE_AMCL_UPDATE})")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Trayectorias (EKF mejorado, sin warnings)")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", type=str, default=str(DEFAULT_BAG), help="Ruta al rosbag (.bag)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.bag))


with AnyReader([bag]) as reader:
    for conn, ts, raw in reader.messages():
        msg = reader.deserialize(raw, conn.msgtype)
        print("TOPIC:", conn.topic)
        print("TIME:", ts * 1e-9)
        print(msg)
        print("=" * 50)
