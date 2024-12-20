## Control System Theory with Python
Slide 1: Introduction to Control System Theory

Control system theory is a fundamental aspect of engineering that deals with the behavior of dynamical systems. It focuses on how systems respond to inputs and how to design controllers to achieve desired outputs. This slide introduces the basic concepts of control systems using a simple temperature control example.

```python
import matplotlib.pyplot as plt
import numpy as np

def simple_thermostat(target_temp, current_temp, time):
    temps = [current_temp]
    for t in range(1, time):
        if temps[-1] < target_temp:
            temps.append(temps[-1] + 0.5)  # Heating
        elif temps[-1] > target_temp:
            temps.append(temps[-1] - 0.3)  # Cooling
        else:
            temps.append(temps[-1])  # Maintain temperature
    return temps

target = 22  # Target temperature in Celsius
initial = 18  # Initial temperature
simulation_time = 60  # Time steps

temperature = simple_thermostat(target, initial, simulation_time)

plt.figure(figsize=(10, 6))
plt.plot(range(simulation_time), temperature, label='Actual Temperature')
plt.axhline(y=target, color='r', linestyle='--', label='Target Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Simple Thermostat Control System')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 2: Open-Loop Control Systems

Open-loop control systems operate without feedback. They use a predetermined set of instructions to control the system output. This slide demonstrates an open-loop water tank filling system.

```python
import matplotlib.pyplot as plt
import numpy as np

def open_loop_water_tank(flow_rate, target_volume, time):
    volumes = [0]
    for t in range(1, time):
        if volumes[-1] < target_volume:
            volumes.append(min(volumes[-1] + flow_rate, target_volume))
        else:
            volumes.append(volumes[-1])
    return volumes

flow_rate = 2  # Liters per time unit
target_volume = 50  # Liters
simulation_time = 30  # Time units

tank_volume = open_loop_water_tank(flow_rate, target_volume, simulation_time)

plt.figure(figsize=(10, 6))
plt.plot(range(simulation_time), tank_volume, label='Tank Volume')
plt.axhline(y=target_volume, color='r', linestyle='--', label='Target Volume')
plt.xlabel('Time')
plt.ylabel('Volume (Liters)')
plt.title('Open-Loop Water Tank Filling System')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 3: Closed-Loop Control Systems

Closed-loop control systems use feedback to adjust the system's output. They continuously compare the actual output with the desired output and make corrections. This slide illustrates a closed-loop cruise control system.

```python
import matplotlib.pyplot as plt
import numpy as np

def closed_loop_cruise_control(target_speed, initial_speed, time, Kp=0.5):
    speeds = [initial_speed]
    for t in range(1, time):
        error = target_speed - speeds[-1]
        acceleration = Kp * error
        new_speed = speeds[-1] + acceleration
        speeds.append(new_speed)
    return speeds

target_speed = 100  # km/h
initial_speed = 80  # km/h
simulation_time = 50  # Time steps

vehicle_speed = closed_loop_cruise_control(target_speed, initial_speed, simulation_time)

plt.figure(figsize=(10, 6))
plt.plot(range(simulation_time), vehicle_speed, label='Vehicle Speed')
plt.axhline(y=target_speed, color='r', linestyle='--', label='Target Speed')
plt.xlabel('Time')
plt.ylabel('Speed (km/h)')
plt.title('Closed-Loop Cruise Control System')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 4: Transfer Functions

Transfer functions are mathematical representations of the relationship between the input and output of a linear time-invariant system. This slide demonstrates how to create and analyze a simple transfer function using Python.

```python
import control
import matplotlib.pyplot as plt

# Define a simple transfer function: G(s) = 1 / (s + 1)
num = [1]  # Numerator coefficients
den = [1, 1]  # Denominator coefficients
sys = control.TransferFunction(num, den)

# Print the transfer function
print("Transfer Function:")
print(sys)

# Generate the step response
t, y = control.step_response(sys)

# Plot the step response
plt.figure(figsize=(10, 6))
plt.plot(t, y)
plt.title('Step Response of G(s) = 1 / (s + 1)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

Slide 5: PID Controllers

PID (Proportional-Integral-Derivative) controllers are widely used in industrial control systems. They combine three control actions to minimize the error between the desired setpoint and the measured process variable. This slide implements a simple PID controller for a temperature control system.

```python
import numpy as np
import matplotlib.pyplot as plt

def pid_controller(Kp, Ki, Kd, setpoint, process_variable, dt, time):
    error_sum = 0
    last_error = 0
    output = []
    errors = []

    for t in np.arange(0, time, dt):
        error = setpoint - process_variable[-1]
        error_sum += error * dt
        error_rate = (error - last_error) / dt

        control_output = Kp * error + Ki * error_sum + Kd * error_rate
        process_variable.append(process_variable[-1] + control_output * dt)
        
        output.append(control_output)
        errors.append(error)
        last_error = error

    return output, errors, process_variable[1:]

# PID parameters
Kp, Ki, Kd = 0.5, 0.1, 0.05
setpoint = 50
initial_temp = 25
dt = 0.1
time = 100

process_variable = [initial_temp]
output, errors, temp = pid_controller(Kp, Ki, Kd, setpoint, process_variable, dt, time)

t = np.arange(0, time, dt)
plt.figure(figsize=(12, 8))
plt.plot(t, temp, label='Temperature')
plt.plot(t, [setpoint]*len(t), '--', label='Setpoint')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('PID Controller for Temperature Control')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: State-Space Representation

State-space representation is a mathematical model of a physical system as a set of input, output, and state variables related by first-order differential equations. This slide demonstrates how to create and analyze a state-space model using Python.

```python
import control
import matplotlib.pyplot as plt
import numpy as np

# Define the state-space matrices for a simple mass-spring-damper system
m = 1.0  # mass
k = 1.0  # spring constant
b = 0.5  # damping coefficient

A = np.array([[0, 1], [-k/m, -b/m]])
B = np.array([[0], [1/m]])
C = np.array([1, 0])
D = np.array([0])

# Create the state-space system
sys = control.StateSpace(A, B, C, D)

# Generate the step response
t, y = control.step_response(sys)

# Plot the step response
plt.figure(figsize=(10, 6))
plt.plot(t, y)
plt.title('Step Response of Mass-Spring-Damper System')
plt.xlabel('Time')
plt.ylabel('Position')
plt.grid(True)
plt.show()

# Print the state-space matrices
print("A matrix:")
print(A)
print("\nB matrix:")
print(B)
print("\nC matrix:")
print(C)
print("\nD matrix:")
print(D)
```

Slide 7: Stability Analysis

Stability is a crucial concept in control system theory. A system is considered stable if it returns to its equilibrium state after a disturbance. This slide demonstrates how to analyze the stability of a system using the roots of its characteristic equation.

```python
import numpy as np
import control
import matplotlib.pyplot as plt

# Define a transfer function: G(s) = (s + 2) / (s^2 + 3s + 2)
num = [1, 2]
den = [1, 3, 2]
sys = control.TransferFunction(num, den)

# Calculate the poles of the system
poles = control.pole(sys)

# Plot the pole-zero map
plt.figure(figsize=(8, 8))
control.pzmap(sys, plot=True)
plt.title('Pole-Zero Map')
plt.grid(True)
plt.show()

# Print the poles
print("System poles:")
for pole in poles:
    print(f"{pole:.3f}")

# Check stability
if all(pole.real < 0 for pole in poles):
    print("The system is stable.")
else:
    print("The system is unstable.")
```

Slide 8: Frequency Response Analysis

Frequency response analysis is a method to characterize how a system responds to different input frequencies. This slide demonstrates how to generate and interpret Bode plots for a given transfer function.

```python
import control
import matplotlib.pyplot as plt

# Define a transfer function: G(s) = 100 / (s^2 + 10s + 100)
num = [100]
den = [1, 10, 100]
sys = control.TransferFunction(num, den)

# Generate Bode plot
plt.figure(figsize=(12, 8))
mag, phase, omega = control.bode(sys, dB=True, Hz=True, plot=True)

# Customize the plot
plt.suptitle('Bode Plot of G(s) = 100 / (s^2 + 10s + 100)')
plt.subplot(211)
plt.title('Magnitude Response')
plt.ylabel('Magnitude (dB)')
plt.grid(True)

plt.subplot(212)
plt.title('Phase Response')
plt.ylabel('Phase (degrees)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print some key features
bandwidth = control.bandwidth(sys)
print(f"System bandwidth: {bandwidth:.2f} rad/s")

gain_margin, phase_margin, _, _ = control.margin(sys)
print(f"Gain margin: {gain_margin:.2f}")
print(f"Phase margin: {phase_margin:.2f} degrees")
```

Slide 9: Root Locus Analysis

Root locus analysis is a graphical method for examining how the roots of a system change with variation of a certain system parameter. This slide demonstrates how to generate and interpret a root locus plot.

```python
import control
import matplotlib.pyplot as plt

# Define a transfer function: G(s) = K / (s(s + 2)(s + 4))
num = [1]
den = [1, 6, 8, 0]
sys = control.TransferFunction(num, den)

# Generate root locus plot
plt.figure(figsize=(10, 8))
rlist, klist = control.root_locus(sys, plot=True)

plt.title('Root Locus of G(s) = K / (s(s + 2)(s + 4))')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.grid(True)

# Add annotations
plt.annotate('Increasing K', xy=(-1, 2), xytext=(-3, 3),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

# Print some key points
print("Open-loop poles:")
print(control.pole(sys))

print("\nOpen-loop zeros:")
print(control.zero(sys))
```

Slide 10: Discrete-Time Systems

Discrete-time systems operate on sampled data, as opposed to continuous-time systems. This slide introduces the concept of discrete-time systems and demonstrates how to analyze them using Python.

```python
import control
import matplotlib.pyplot as plt
import numpy as np

# Define a discrete-time transfer function: G(z) = 0.1z / (z - 0.9)
num = [0.1, 0]
den = [1, -0.9]
dt = 0.1  # Sample time
sys_d = control.TransferFunction(num, den, dt)

# Generate step response
t, y = control.step_response(sys_d)

# Plot step response
plt.figure(figsize=(10, 6))
plt.step(t, y, where='post')
plt.title('Step Response of Discrete-Time System G(z) = 0.1z / (z - 0.9)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Print system information
print("Discrete-time transfer function:")
print(sys_d)

print("\nPoles:")
print(control.pole(sys_d))

print("\nZeros:")
print(control.zero(sys_d))

# Generate impulse response
t_imp, y_imp = control.impulse_response(sys_d)

# Plot impulse response
plt.figure(figsize=(10, 6))
plt.stem(t_imp, y_imp)
plt.title('Impulse Response of Discrete-Time System')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

Slide 11: State Feedback Control

State feedback control is a method where the full state of the system is used to determine the control input. This slide demonstrates how to design a state feedback controller for a simple system.

```python
import control
import numpy as np
import matplotlib.pyplot as plt

# Define the system matrices
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([1, 0])
D = np.array([0])

# Create the state-space system
sys = control.StateSpace(A, B, C, D)

# Design the state feedback gain
desired_poles = [-2, -3]
K = control.place(A, B, desired_poles)

# Create the closed-loop system
A_cl = A - np.dot(B, K)
sys_cl = control.StateSpace(A_cl, B, C, D)

# Generate step response for both open-loop and closed-loop systems
t, y_ol = control.step_response(sys)
t, y_cl = control.step_response(sys_cl)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, y_ol, label='Open-loop')
plt.plot(t, y_cl, label='Closed-loop')
plt.title('Step Response: Open-loop vs. Closed-loop')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

print("State feedback gain K:")
print(K)

print("\nClosed-loop poles:")
print(control.pole(sys_cl))
```

Slide 12: Observability and Controllability

Observability and controllability are important properties of control systems. A system is observable if its internal states can be inferred from its outputs, and controllable if it can be driven to any desired state. This slide demonstrates how to check these properties for a given system.

```python
import numpy as np
import control

# Define the system matrices
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([1, 0])
D = np.array([0])

# Create the state-space system
sys = control.StateSpace(A, B, C, D)

# Check controllability
Wc = control.ctrb(A, B)
rank_Wc = np.linalg.matrix_rank(Wc)
n = A.shape[0]  # System order

if rank_Wc == n:
    print("The system is controllable")
else:
    print("The system is not controllable")

# Check observability
Wo = control.obsv(A, C)
rank_Wo = np.linalg.matrix_rank(Wo)

if rank_Wo == n:
    print("The system is observable")
else:
    print("The system is not observable")

# Print controllability and observability matrices
print("\nControllability matrix:")
print(Wc)
print("\nObservability matrix:")
print(Wo)
```

Slide 13: Kalman Filter

The Kalman filter is an algorithm that uses a series of measurements observed over time, containing statistical noise and other inaccuracies, and produces estimates of unknown variables that tend to be more accurate than those based on a single measurement alone. This slide demonstrates a simple implementation of a Kalman filter.

```python
import numpy as np
import matplotlib.pyplot as plt

def kalman_filter(z, x, P, F, H, R, Q):
    # Predict
    x = np.dot(F, x)
    P = np.dot(np.dot(F, P), F.T) + Q

    # Update
    y = z - np.dot(H, x)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    x = x + np.dot(K, y)
    P = P - np.dot(np.dot(K, H), P)

    return x, P

# System parameters
F = np.array([[1, 1], [0, 1]])  # State transition matrix
H = np.array([[1, 0]])  # Measurement matrix
Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise covariance
R = np.array([[1]])  # Measurement noise covariance

# Initial state
x = np.array([[0], [1]])
P = np.eye(2)

# Generate noisy measurements
true_positions = np.linspace(0, 100, 100)
measurements = true_positions + np.random.normal(0, 10, 100)

# Apply Kalman filter
filtered_positions = []
for z in measurements:
    x, P = kalman_filter(z, x, P, F, H, R, Q)
    filtered_positions.append(x[0, 0])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(true_positions, label='True Position')
plt.plot(measurements, 'r.', label='Noisy Measurements')
plt.plot(filtered_positions, 'g-', label='Kalman Filter Estimate')
plt.legend()
plt.title('Kalman Filter for Position Estimation')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.grid(True)
plt.show()
```

Slide 14: Model Predictive Control (MPC)

Model Predictive Control is an advanced method of process control that uses a model of the process to predict future behavior and optimize control actions. This slide presents a simplified example of MPC for a basic system.

```python
import numpy as np
import matplotlib.pyplot as plt

def simple_mpc(x0, setpoint, horizon, A, B):
    x = x0
    trajectory = [x]
    control_inputs = []

    for _ in range(horizon):
        # Simple optimization: move towards setpoint
        u = (setpoint - x) * 0.1
        
        # Apply control input
        x = A * x + B * u
        
        trajectory.append(x)
        control_inputs.append(u)

    return trajectory, control_inputs

# System parameters
A = 0.9  # System dynamics
B = 0.1  # Input effect

# MPC parameters
x0 = 0  # Initial state
setpoint = 10  # Desired setpoint
horizon = 50  # Prediction horizon

# Run MPC
trajectory, control_inputs = simple_mpc(x0, setpoint, horizon, A, B)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(horizon + 1), trajectory, label='System State')
plt.plot([0, horizon], [setpoint, setpoint], 'r--', label='Setpoint')
plt.step(range(horizon), control_inputs, label='Control Input')
plt.legend()
plt.title('Simple Model Predictive Control')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

For further exploration of control system theory, consider the following resources:

1. "Modern Control Engineering" by Katsuhiko Ogata
2. "Control Systems Engineering" by Norman S. Nise
3. ArXiv.org Control Systems Category: [https://arxiv.org/list/eess.SY/recent](https://arxiv.org/list/eess.SY/recent)
4. Python Control Systems Library Documentation: [https://python-control.readthedocs.io/](https://python-control.readthedocs.io/)
5. MATLAB Control Systems Toolbox (for those with access to MATLAB)

These resources provide in-depth coverage of control system theory, from fundamental concepts to advanced techniques. The ArXiv link offers access to recent research papers in the field, while the Python Control Systems Library documentation is an excellent resource for implementing control systems in Python.

