import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

def main():
    # --- Configuration & Initial State ---
    omega = 1.0        # Angular frequency
    T_max = 4 * np.pi  # Show two full periods of reference cos(wt)
    num_points = 1000
    t = np.linspace(0, T_max, num_points)
    
    # Internal state stored as complex number A
    # Initial state: A = 1 + 1j (Amp=sqrt(2), Phase=45 deg)
    state = {'A': complex(1.0, 1.0)}
    
    # Flag to prevent recursion during synchronization
    is_updating_ui = [False] 

    # --- Setup Figure and Axes ---
    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.9)
    ax = fig.add_subplot(111)
    
    # Reference Signal: cos(wt)
    sig_ref = np.cos(omega * t)
    line_ref, = ax.plot(t, sig_ref, lw=1.5, color='gray', linestyle='--', label=r'$\cos(\omega t)$')
    
    # Target Signal: Re(A * e^(iwt))
    # We use R*cos(wt + phi) equivalent form for easier calculation
    line_target, = ax.plot(t, np.zeros(num_points), lw=2.5, color='royalblue', label=r'$\text{Re}(A e^{i\omega t})$')
    
    # Formatting
    ax.set_title(r'Comparison: Reference $\cos(\omega t)$ vs. $\text{Re}(A e^{i\omega t})$', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_xlim(0, T_max)
    
    # Dynamic grid to show pi multiples on x-axis
    ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
    
    ax.axhline(0, color='black', lw=1)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Legend placed in upper right, ignoring reference for clean view if needed
    leg = ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)

    # --- Mathematical Update Logic ---
    def update_plot():
        """Recalculates signals based on current internal state and updates lines."""
        A = state['A']
        
        # Method 1: Direct definition Re((a+ib)*(cos+isin)) -> a*cos - b*sin
        # sig_target = A.real * np.cos(omega * t) - A.imag * np.sin(omega * t)
        
        # Method 2: Polar form Re(R*e^iphi * e^iwt) -> R*cos(wt + phi)
        R = np.abs(A)
        phi = np.angle(A) # radians
        sig_target = R * np.cos(omega * t + phi)
        
        line_target.set_ydata(sig_target)
        
        # Dynamic Y-limit scaling based on amplitude, min [-1.5, 1.5]
        limit = max(1.5, R * 1.1)
        ax.set_ylim(-limit, limit)
        
        # Update title text with current values (Phase in degrees for readability)
        phi_deg = np.degrees(phi)
        ax.set_title(r'Comparison: $\cos(\omega t)$ vs. $\text{Re}([%.2f + %.2fi] e^{i\omega t})$' % (A.real, A.imag) + \
                     '\n' + r'Polar: $A = %.2f e^{i(%.1f^\circ)}$' % (R, phi_deg), fontsize=12)
        
        fig.canvas.draw_idle()

    # --- UI Synchronization Logic ---
    def synchronize_ui_inputs(source):
        """
        Updates textboxes to match internal state without triggering 
        submit events recursively.
        """
        if is_updating_ui[0]: return # Stop recursion
        is_updating_ui[0] = True
        
        A = state['A']
        R = np.abs(A)
        phi_deg = np.degrees(np.angle(A))
        
        if source != 'cartesian':
            txt_real.set_val("%.3f" % A.real)
            txt_imag.set_val("%.3f" % A.imag)
            
        if source != 'polar':
            txt_amp.set_val("%.3f" % R)
            txt_phase.set_val("%.1f" % phi_deg)
            
        is_updating_ui[0] = False

    # --- Textbox Callbacks ---
    def submit_cartesian(text):
        if is_updating_ui[0]: return
        try:
            # Read both Cartesian boxes, combine into new state
            re = float(txt_real.text)
            im = float(txt_imag.text)
            state['A'] = complex(re, im)
            
            # Update plot and synchronize the Polar boxes
            update_plot()
            synchronize_ui_inputs(source='cartesian')
        except ValueError:
            pass # Ignore invalid inputs (empty strings, text)

    def submit_polar(text):
        if is_updating_ui[0]: return
        try:
            # Read both Polar boxes
            R = float(txt_amp.text)
            # Ensure Amplitude isn't negative for standard polar form
            if R < 0: R = 0; txt_amp.set_val("0.000") 
            
            phi_deg = float(txt_phase.text)
            phi_rad = np.radians(phi_deg)
            
            # Convert Polar -> Complex Cartesian state: R * e^(i*phi)
            state['A'] = R * np.exp(1j * phi_rad)
            
            # Update plot and synchronize the Cartesian boxes
            update_plot()
            synchronize_ui_inputs(source='polar')
        except ValueError:
            pass # Ignore invalid inputs

    # --- Create UI Layout ---
    # Define areas for textboxes at bottom of figure [left, bottom, width, height]
    y_row1 = 0.10
    y_row2 = 0.04
    box_w = 0.12
    box_h = 0.04
    label_pad = 0.08 # Space between groups

    # Row 1: Cartesian
    ax_real  = plt.axes([0.15, y_row1, box_w, box_h])
    ax_imag  = plt.axes([0.15 + box_w + label_pad, y_row1, box_w, box_h])
    
    # Row 2: Polar
    ax_amp   = plt.axes([0.15, y_row2, box_w, box_h])
    ax_phase = plt.axes([0.15 + box_w + label_pad, y_row2, box_w, box_h])

    # Initialize TextBoxes with default state values
    A_init = state['A']
    txt_real = TextBox(ax_real, r'Real Part $\text{Re}(A)$: ', valinit="%.3f" % A_init.real)
    txt_imag = TextBox(ax_imag, r'Imag Part $\text{Im}(A)$: ', valinit="%.3f" % A_init.imag)
    
    R_init = np.abs(A_init)
    phi_init_deg = np.degrees(np.angle(A_init))
    txt_amp   = TextBox(ax_amp, r'Amplitude $|A|$: ', valinit="%.3f" % R_init)
    txt_phase = TextBox(ax_phase, r'Phase $\phi$ (deg): ', valinit="%.1f" % phi_init_deg)

    # Attach callbacks triggered when user hits 'Enter' in a box
    txt_real.on_submit(submit_cartesian)
    txt_imag.on_submit(submit_cartesian)
    txt_amp.on_submit(submit_polar)
    txt_phase.on_submit(submit_polar)

    # --- Finalize ---
    update_plot() # Initial plot draw
    plt.show()

if __name__ == "__main__":
    main()