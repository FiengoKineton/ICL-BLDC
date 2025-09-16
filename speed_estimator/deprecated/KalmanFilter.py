import numpy as np

# -----------------------------------------------------------------------------
# Basic (Extended) Kalman filter utilities for linear time-varying outputs.
# Notation:
#   x_t ∈ ℝ^{n} : state estimate at time t
#   P_t ∈ ℝ^{n×n} : state covariance
#   F ∈ ℝ^{n×n} : state transition (assumed time-invariant here)
#   Q ∈ ℝ^{n×n} or scalar : process noise covariance
#   H_t ∈ ℝ^{m×n} : output matrix at time t (taken row-wise from output_matrix)
#   R or R_t ∈ ℝ^{m×m} : measurement noise covariance
#   z_t ∈ ℝ^{m} : measurement at time t
# Equations:
#   Predict:
#     x_t^- = F x_{t-1}
#     P_t^- = F P_{t-1} Fᵀ + Q
#   Update:
#     S_t = H_t P_t^- H_tᵀ + R
#     K_t = P_t^- H_tᵀ S_t^{-1}
#     x_t = x_t^- + K_t (z_t - H_t x_t^-)
#     P_t = (I - K_t H_t) P_t^-     # (Joseph form recommended for stability)
# -----------------------------------------------------------------------------


#KALMAN FILTER FUNCTION DEFINITION

def kf_predict(F, Q, H, x_prev, P_prev):
    """Kalman predict step (linear).
    Args:
        F: (n×n) state transition
        Q: process noise covariance (n×n) or scalar
        H: (m×n) output matrix (only used here for the custom Q scaling line below)
        x_prev: (n×1) previous state estimate
        P_prev: (n×n) previous covariance
    Returns:
        x_t: predicted state (n×1)
        y_t: predicted output H x_t (m×1)
        P_t: predicted covariance
    NOTE:
        The line 'P_t = F P_prev Fᵀ + Q * (H > 0)' is non-standard. In vanilla KF:
            P_t = F P_prev Fᵀ + Q
        Multiplying Q by (H > 0) gates process noise by measurement structure, which
        is unconventional and may lead to under/over-estimation of uncertainty.
        Use with caution unless this is an intentional heuristic.
    """
    # Perform prediction step
    x_t = F @ x_prev

    # Predict output
    y_t = H @ x_t

    # WARNING: Non-standard covariance update.
    # Standard KF: P_t = F @ P_prev @ F.T + Q
    # Here:        P_t = F @ P_prev @ F.T + Q * (H > 0)
    # '(H > 0)' acts as a mask with shape of H; ensure broadcasting is as intended.
    P_t = F @ P_prev @ F.T + Q * (H > 0)

    return x_t, y_t, P_t

def kf_update(H, R, z_t, x_t, P_t):
    """Kalman update step (linear measurement).
    Args:
        H: (m×n) measurement matrix at time t
        R: (m×m) measurement noise covariance
        z_t: (m×1) measurement at time t
        x_t: (n×1) predicted state from predict step
        P_t: (n×n) predicted covariance from predict step
    Returns:
        x_t: updated state (n×1)
        P_t: updated covariance (n×n)
        K_t: Kalman gain (n×m)
    Notes:
        - S_t must be positive-definite; if ill-conditioned, consider adding jitter to R.
        - For numerical stability, consider Joseph form:
            P_t = (I - K H) P (I - K H)ᵀ + K R Kᵀ
    """
    # Perform update step
    S_t = H @ P_t @ H.T + R
    K_t = P_t @ H.T @ np.linalg.inv(S_t)
    
    # Update state estimate
    x_t = x_t + K_t @ (z_t - H @ x_t)

    # x_t = np.maximum(x_t, 0)

    # NOTE: Covariance update uses simple form: P = (I - K H) P.
    # More stable alternative (Joseph form):
    # I = np.eye(P_t.shape[0])
    # P_t = (I - K_t @ H) @ P_t @ (I - K_t @ H).T + K_t @ R @ K_t.T
    P_t = (np.eye(x_t.shape[0]) - K_t @ H) @ P_t

    return x_t, P_t, K_t

def my_kf(x_0, P_0, F, Q, R, output_matrix, measurements):
    """Run KF with time-varying output matrix H_t (row-wise from output_matrix).
    Args:
        x_0: (n×1) initial state
        P_0: (n×n) initial covariance
        F, Q, R: standard KF matrices (R can be (m×m), constant over time)
        output_matrix: (T×n) where each row is H_t for a single-output case (m=1),
                       or (T×m×n) if extended (code assumes row slicing -> m=1).
        measurements: (T×1) or (T,) sequence of z_t
    Returns:
        states:  (T+1, n, 1) stacked x_t (includes initial x_0 at index 0)
        outputs: (T, 1) stacked y_t = H_t x_t^- (predicted before update)
        cov_matrix: (T+1, n, n) stacked P_t
        k_s: (T, n) gain columns for m=1 case (K_t[:, 0])
    Notes:
        - Appends predicted output y_t before the measurement update; this matches
          "innovation" usage but be aware y_t ≠ H_t x_t (after update).
        - Assumes single-output (m=1) due to K_t[:,0] indexing.
    """
    states = []     # x hat
    outputs = []    # y hat
    cov_matrix = []  # P
    k_s = []

    for t in range(output_matrix.shape[0]):
        H = output_matrix[t:t+1, :]        # H_t as a (1×n) row for scalar measurement; adapt slicing if m>1.
        if t == 0:
            states.append(x_0)
            cov_matrix.append(P_0)

        # Prediction step
        x_t, y_t, P_t = kf_predict(F=F, Q=Q, H=H, x_prev=states[t], P_prev=cov_matrix[t])
        # Update step
        x_t, P_t, K_t = kf_update(H, R, measurements[t], x_t, P_t)
        
        # Append results
        states.append(x_t)
        outputs.append(y_t[0])        # Store predicted output y_t (pre-update). If you want post-update y, use H @ x_t instead.
        cov_matrix.append(P_t)
        k_s.append(K_t[:, 0])

    states = np.array(states)
    outputs = np.array(outputs)
    
    return states, outputs, np.array(cov_matrix), np.array(k_s)

def my_var_kf(x_0, P_0, F, Q, R, output_matrix, measurements, x_hat):
    """Variant KF with ad-hoc variance floor using external reference x_hat.
    Args:
        x_0, P_0, F, Q, R, output_matrix, measurements: as in my_kf
        x_hat: (T×n) external reference trajectory (e.g., prior means) used to derive a lower bound
    Returns:
        states, outputs, cov_matrix, k_s: as in my_kf
    WARNING:
        - The code computes 'var_x_pred = x_hat[t] - 2*sqrt(x_t)' and then adjusts x_t entries
          when this bound is negative. This is NOT a standard statistical variance handling:
              * x_t is a state mean vector, not a variance.
              * Taking sqrt(x_t) assumes positivity and variance-like semantics.
          Use only if this heuristic is intentional; otherwise consider constraining P_t (covariance),
          not x_t (mean), e.g., via Joseph form or eigenvalue flooring on P_t.
    """
    states = []     # x hat
    outputs = []    # y hat
    cov_matrix = []  # P
    k_s = []


    for t in range(output_matrix.shape[0]):
        H = output_matrix[t:t+1, :]
        if t == 0:
            states.append(x_0)
            cov_matrix.append(P_0)

        # Prediction step
        x_t, y_t, P_t = kf_predict(F=F, Q=Q, H=H, x_prev=states[t], P_prev=cov_matrix[t])

        # Update step
        x_t, P_t, K_t = kf_update(H, R, measurements[t], x_t, P_t)

        # Compute the lower confidence bound
        var_x_pred = x_hat[t] - 2 * np.sqrt(x_t)  

        # Adjust variances where the lower bound is negative
        # Heuristic floor on "variance-like" quantity using x_hat; non-standard.
        # Prefer modifying P_t (covariance) if you need variance floors.
        for i in range(len(x_t)):
            if var_x_pred[i] < 0:  # Lower bound is invalid
                x_t[i] = (x_hat[t][i] / 2) ** 2  # Set minimum variance to satisfy the constraint
        
        # Append results
        states.append(x_t)
        outputs.append(y_t[0])
        cov_matrix.append(P_t)
        k_s.append(K_t[:, 0])

    states = np.array(states)
    outputs = np.array(outputs)

    # # Optional clipping for non-negative states
    # states[states < 0] = 0
    # outputs[outputs < 0] = 0
    
    return states, outputs, np.array(cov_matrix), np.array(k_s)

def my_akf(x_0, P_0, F, Q, Rs, output_matrix, measurements):
    """Adaptive KF with time-varying measurement noise R_t.
    Args:
        x_0, P_0, F, Q: as usual
        Rs: sequence/list of R_t (m×m) per time step
        output_matrix: provides H_t per time step
        measurements: sequence of z_t
    Returns:
        states, outputs, cov_matrix: stacked arrays as in my_kf
    Notes:
        - Useful when SNR changes over time; e.g., low-speed BLDC regimes with weaker back-EMF.
        - Keep Rs[t] positive-definite to avoid S_t inversion issues.
    """
    states = []     #x hat
    outputs = []    # y hat
    cov_matrix = []  # P

    for t in range(output_matrix.shape[0]):
        H = output_matrix[t:t+1,:]
        R = Rs[t]
        if t == 0:
            states.append(x_0)
            cov_matrix.append(P_0)

        x_t,y_t,P_t = kf_predict(F=F,Q=Q,H=H,x_prev=states[t],P_prev=cov_matrix[t])
        x_t,P_t,K_t = kf_update(H,R,measurements[t],x_t,P_t)

        states.append(x_t)    
        outputs.append(y_t[0])
        cov_matrix.append(P_t)
    
    return np.array(states),np.array(outputs),np.array(cov_matrix)



# GOTCHA:
# - kf_predict uses a non-standard P update with Q*(H>0); replace with '+ Q' for classical KF.
# - outputs store pre-update predictions y_t = H x_t^-; if you want filtered output, compute H x_t after update.
# - Code assumes scalar measurement (m=1) because it indexes K_t[:,0] and y_t[0]; generalize for m>1.
# - For numerical stability, prefer Joseph form for covariance update and add small jitter to R if S_t is near-singular.
# - Ensure shapes: measurements[t] must broadcast to (m,1) or (m,), H as (m×n).
