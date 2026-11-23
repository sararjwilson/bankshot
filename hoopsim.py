import numpy as np
from dataclasses import dataclass

# geometry / physical constants
@dataclass
class Court:
    R: float = 0.12            # ball radius (nba size ~ 0.119 m)
    rim_y: float = 3.048       # 10 ft
    rim_inner_d: float = 0.457 # 18 in inner diameter
    rim_tube: float = 0.008    # radius of the rim's metal tubing
    board_gap: float = 0.152   # 6 in from backboard face to inside of back rim
    board_bottom: float = 2.90
    board_top: float = 3.97
    g: float = 9.81
    drag_k: float = 0.022      # 0.5*rho*Cd*A/m  (1/m)
    e_board: float = 0.75      # restitution, backboard
    mu_board: float = 0.35     # friction, backboard
    e_rim: float = 0.60        # restitution, rim
    mu_rim: float = 0.30       # friction, rim
    inertia_factor: float = 2.0 / 3.0   # I = f m R^2  (basketball is a hollow sphere)

    @property
    def back_rim_x(self):
        # center of the back rim tube
        return -(self.board_gap + self.rim_tube)

    @property
    def front_rim_x(self):
        # center of the front rim tube
        return self.back_rim_x - (self.rim_inner_d + 2 * self.rim_tube)

    @property
    def rim_centers(self):
        return [(self.front_rim_x, self.rim_y), (self.back_rim_x, self.rim_y)]

# collision impulse

def _collide(vx, vy, w, nx, ny, e, mu, R, f):
    """
    apply a collision impulse to balls whose contact normal (pointing from
    the obstacle toward the ball center) is (nx, ny).  mass is 1.

    returns new (vx, vy, w).
    """
    # contact point relative to ball center: r_c = -R n
    # velocity of contact point: v + w z_hat x r_c = v + w * R * (n_y, -n_x)
    tx, ty = -ny, nx                          # unit tangent
    vcx = vx + w * R * ny
    vcy = vy - w * R * nx
    vn = vcx * nx + vcy * ny                  # normal
    vt = vcx * tx + vcy * ty                  # slip

    approaching = vn < 0
    Jn = np.where(approaching, -(1.0 + e) * vn, 0.0)

    # tangential impulse that would exactly stop slip
    #   dv_t = Jt * (1/m + R^2/I) = Jt * (1 + 1/f)
    Jt_stick = -vt / (1.0 + 1.0 / f)
    Jt = np.clip(Jt_stick, -mu * Jn, mu * Jn)

    vx = vx + Jn * nx + Jt * tx
    vy = vy + Jn * ny + Jt * ty
    w = w - R * Jt / (f * R * R)              # dw = -R Jt / I
    return vx, vy, w

# simmy
# what the ball touched first
FIRST_NONE, FIRST_BOARD, FIRST_RIM = 0, 1, 2


def simulate(x0, y0, vx0, vy0, w0, court=Court(), dt=2e-4, t_max=3.0,
             record_paths=False, path_stride=25):
    """
    sim N shots in parallel.

    params are 1d arrays of length N (release position, velocity, spin).

    returns a dict of length-N arrays:
        made          1 if the ball went through the hoop, else 0
        first_contact FIRST_NONE / FIRST_BOARD / FIRST_RIM
        impact_y      height of ball center at first backboard contact (nan if none)
        impact_vx/vy  ball velocity just BEFORE first backboard contact
        impact_w      ball spin just before first backboard contact
        n_rim_hits    number of rim collisions
    if record_paths, a list of (k, 2) arrays with sampled trajectories.
    """
    c = court
    R, f = c.R, c.inertia_factor
    x = np.asarray(x0, float).copy()
    y = np.asarray(y0, float).copy()
    vx = np.asarray(vx0, float).copy()
    vy = np.asarray(vy0, float).copy()
    w = np.asarray(w0, float).copy()
    N = x.size

    made = np.zeros(N, int)
    first = np.full(N, FIRST_NONE)
    imp_y = np.full(N, np.nan)
    imp_vx = np.full(N, np.nan)
    imp_vy = np.full(N, np.nan)
    imp_w = np.full(N, np.nan)
    n_rim = np.zeros(N, int)
    # a ball is "in contact" while overlapping an obstacle; we only apply one impulse per contact episode
    in_board = np.zeros(N, bool)
    in_rim = np.zeros((2, N), bool)

    active = np.ones(N, bool)
    paths = [[(x[i], y[i])] for i in range(N)] if record_paths else None

    n_steps = int(t_max / dt)
    for step in range(n_steps):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        ax_, ay_ = x[idx], y[idx]
        bvx, bvy, bw = vx[idx], vy[idx], w[idx]
        prev_y = ay_.copy()

        # free flight: semi-implicit Euler with drag
        speed = np.hypot(bvx, bvy)
        bvx = bvx - c.drag_k * speed * bvx * dt
        bvy = bvy - (c.g + c.drag_k * speed * bvy) * dt
        ax_ = ax_ + bvx * dt
        ay_ = ay_ + bvy * dt

        # backboard (vertical segment at x = 0)
        on_board_height = (ay_ >= c.board_bottom) & (ay_ <= c.board_top)
        touching = on_board_height & (ax_ + R > 0.0)
        new_hit = touching & ~in_board[idx] & (bvx > 0)
        if new_hit.any():
            j = idx[new_hit]
            # record pre-impact state on the FIRST contact of any kind
            firsthit = first[j] == FIRST_NONE
            jj = j[firsthit]
            first[jj] = FIRST_BOARD
            imp_y[jj] = ay_[new_hit][firsthit]
            imp_vx[jj] = bvx[new_hit][firsthit]
            imp_vy[jj] = bvy[new_hit][firsthit]
            imp_w[jj] = bw[new_hit][firsthit]

            nvx, nvy, nw = _collide(bvx[new_hit], bvy[new_hit], bw[new_hit],
                                    -1.0, 0.0, c.e_board, c.mu_board, R, f)
            bvx[new_hit], bvy[new_hit], bw[new_hit] = nvx, nvy, nw
            ax_[new_hit] = -R            # push out of the board
        in_board[idx] = touching

        # rim edges (two small circles)
        for k, (cx, cy) in enumerate(c.rim_centers):
            dx, dy = ax_ - cx, ay_ - cy
            dist = np.hypot(dx, dy)
            touching = dist < (R + c.rim_tube)
            if touching.any():
                nx = np.where(touching, dx / np.maximum(dist, 1e-12), 0.0)
                ny = np.where(touching, dy / np.maximum(dist, 1e-12), 0.0)
                new_hit = touching & ~in_rim[k, idx]
                if new_hit.any():
                    j = idx[new_hit]
                    first[j[first[j] == FIRST_NONE]] = FIRST_RIM
                    n_rim[j] += 1
                    nvx, nvy, nw = _collide(bvx[new_hit], bvy[new_hit], bw[new_hit],
                                            nx[new_hit], ny[new_hit],
                                            c.e_rim, c.mu_rim, R, f)
                    bvx[new_hit], bvy[new_hit], bw[new_hit] = nvx, nvy, nw
                # push overlapping balls out to the surface
                push = touching
                ax_[push] = cx + nx[push] * (R + c.rim_tube)
                ay_[push] = cy + ny[push] * (R + c.rim_tube)
            in_rim[k, idx] = touching

        # scoring: center passes downward through rim plane between rims
        through = ((prev_y >= c.rim_y) & (ay_ < c.rim_y)
                   & (ax_ > c.front_rim_x) & (ax_ < c.back_rim_x))
        made[idx[through]] = 1

        # write back and retire finished shots
        x[idx], y[idx], vx[idx], vy[idx], w[idx] = ax_, ay_, bvx, bvy, bw
        done = through | ((ay_ < c.rim_y - 0.5) & (bvy < 0)) | (ax_ < -12)
        active[idx[done]] = False

        if record_paths and step % path_stride == 0:
            for i in idx:
                paths[i].append((x[i], y[i]))

    out = dict(made=made, first_contact=first, impact_y=imp_y,
               impact_vx=imp_vx, impact_vy=imp_vy, impact_w=imp_w,
               n_rim_hits=n_rim)
    if record_paths:
        out["paths"] = [np.array(p) for p in paths]
    return out

# rando gen

def random_bank_shots(n, rng=None, court=Court()):
    """
    sample n shot releases aimed at a random spot on the
    backboard (with noise), so most of them are banks.

    returns release arrays x0, y0, vx0, vy0, w0.
    """
    rng = np.random.default_rng(rng)
    c = court
    dist = rng.uniform(1.5, 6.0, n)               # horizontal distance to board
    x0 = -dist
    y0 = rng.uniform(2.0, 2.5, n)                 # release height
    theta = np.radians(rng.uniform(35, 62, n))    # launch angle

    # aim: ballistic shot that hits the board at height y_t
    y_t = rng.uniform(3.05, 3.65, n)
    dx = dist - c.R
    dy = y_t - y0
    # y = x tan(th) - g x^2 / (2 v^2 cos^2 th)  ->  solve for v
    denom = 2 * np.cos(theta) ** 2 * (dx * np.tan(theta) - dy)
    ok = denom > 0
    v = np.where(ok, np.sqrt(c.g * dx ** 2 / np.where(ok, denom, 1.0)), 0.0)
    v *= rng.normal(1.0, 0.02, n)                 # shooter imperfection
    v *= 1 + c.drag_k * dist * 0.5                # rough drag compensation

    vx0 = v * np.cos(theta)
    vy0 = v * np.sin(theta)
    w0 = rng.uniform(-5, 35, n)                   # mostly backspin, rad/s
    keep = ok & (v > 0)
    return x0[keep], y0[keep], vx0[keep], vy0[keep], w0[keep]
