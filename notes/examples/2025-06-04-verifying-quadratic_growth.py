import numpy as np


def evaluate_expression_on_circle(theta_rad):
    """
    Evaluates the expression x_c - (1/2) * ||x_pt - x_0||^2 for a point x_pt
    on the unit circle, parameterized by the angle theta_rad.

    Args:
        theta_rad (float): The angle in radians that defines the point on the
                           unit circle.

    Returns:
        float: The value of the expression. Expected to be -1.0.
    """
    # The fixed point x_0
    x0 = np.array([-1, 0])

    # Calculate the coordinates of the point x_pt on the unit circle
    xc = np.cos(theta_rad)
    yc = np.sin(theta_rad)
    x_pt = np.array([xc, yc])

    # Calculate the vector difference x_pt - x_0
    diff_vector = x_pt - x0

    # Calculate the squared Euclidean norm of the difference vector
    # norm_sq = np.linalg.norm(diff_vector)**2 is another way
    norm_sq = np.sum(diff_vector**2)

    # Calculate the final expression
    value = xc - 0.5 * norm_sq

    return value


if __name__ == "__main__":
    # --- Example Usage ---
    angles_degrees = [0, 30, 45, 60, 90, 120, 135, 150, 180, 270, 360]
    print("Evaluating the expression for various angles:")
    print("---------------------------------------------")
    print(f"{'Angle (deg)':<12} | {'Angle (rad)':<12} | {'Expression Value':<20}")
    print("---------------------------------------------")

    for angle_deg in angles_degrees:
        angle_rad = np.deg2rad(angle_deg)
        result = evaluate_expression_on_circle(angle_rad)
        # Format output for better readability
        print(f"{angle_deg:<12.1f} | {angle_rad:<12.4f} | {result:<20.15f}")

    print("---------------------------------------------")
    print("\nTesting with some specific numpy values for pi:")

    theta_pi_half = np.pi / 2
    result_pi_half = evaluate_expression_on_circle(theta_pi_half)
    print(f"For theta = pi/2 ({theta_pi_half:.4f} rad), value = {result_pi_half:.15f}")

    theta_pi = np.pi
    result_pi = evaluate_expression_on_circle(theta_pi)
    print(f"For theta = pi ({theta_pi:.4f} rad), value = {result_pi:.15f}")

    theta_2pi = 2 * np.pi
    result_2pi = evaluate_expression_on_circle(theta_2pi)
    print(f"For theta = 2*pi ({theta_2pi:.4f} rad), value = {result_2pi:.15f}")

    # Test an arbitrary angle
    arbitrary_theta = 1.2345  # radians
    result_arbitrary = evaluate_expression_on_circle(arbitrary_theta)
    print(f"For theta = {arbitrary_theta:.4f} rad, value = {result_arbitrary:.15f}")
