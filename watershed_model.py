import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ipywidgets import interact, FloatSlider
from IPython.display import HTML

class WatershedModel:
    def __init__(self):
        # km
        self.Lx = 1
        self.Ly = 0.5
        self.nx = 512
        self.ny = 256
        
        self.nu = 0.01

        self.x = np.linspace(0,self.Lx,self.nx)
        self.y = np.linspace(0,self.Ly,self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.IC = np.exp(-0.5 * (self.Lx/2 - self.X)**2/(0.01*self.Lx)**2 - 0.5 * (self.Ly/2 - self.Y)**2/(0.01*self.Ly)**2 )
        self.IC /= self.get_total_contaminant(self.IC) # scale it so total concentration = 1 before mitigation

        # Saving the IC fourier coeffs with no mitigation
        self.IChat_nm = np.fft.fft2(self.IC)

        ijx = 2 * np.pi * np.fft.fftfreq(self.nx, self.Lx/self.nx)
        iky = 2 * np.pi * np.fft.fftfreq(self.ny, self.Ly/self.ny)
        J, K = np.meshgrid(ijx,iky)
        self.diffusion = -self.nu * (J**2 + K**2)

    def run_simulation(self, t, source_mitigation=0.0, sink_rate=0.0):
        IChat = np.fft.fft2(self.IC*(1-source_mitigation))
        chat_nm = self.IChat_nm * np.exp(t * self.diffusion)
        chat = IChat * np.exp(t * (self.diffusion - sink_rate))
        self.c_nm = np.fft.ifftn(chat_nm).real
        self.c = np.fft.ifftn(chat).real
        return self.c
    
    def get_total_contaminant(self,C=None):
        if C is None: C = self.c
        return np.trapz(np.trapz(C,self.x,axis=1),self.y,axis=0)
    
    def plot_concentration(self,C=None, source_mitigation=None, sink_rate=None):
        if C is None: C = self.c
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.imshow(C, vmin=0, vmax=self.c_nm.max(), cmap='Greys', origin='lower')
        ax.annotate(f"{100*(1-self.get_total_contaminant(C)):.0f}% removed", (0.02,0.97), xycoords='axes fraction', va='top')

        if not source_mitigation is None:
            sm_cost = source_mitigation*1e7
            sr_cost = sink_rate*5e6
            total_cost = sm_cost + sr_cost
            budget=3e6
            ax.annotate(f"Budget: ${budget:,.0f}", (0.02, 0.87), xycoords='axes fraction', va='top')
            ax.annotate("Total cost:", (0.02, 0.78), xycoords='axes fraction', va='top')
            if total_cost <= budget:
                ax.annotate(f"${total_cost:,.0f}", (0.22, 0.78), xycoords='axes fraction', va='top')
            else:
                ax.annotate(f"${total_cost:,.0f}", (0.22, 0.78), xycoords='axes fraction', va='top', color='red')

        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        
    def interactive_simulation(self):
        @interact(source_mitigation=FloatSlider(description="Source removal", min=0, max=1, step=0.1, value=0.0, style=dict(description_width='initial')),
                  sink_rate=FloatSlider(description="Soil treatment", min=0, max=1, step=0.1, value=0.0, style=dict(description_width='initial')))
        def plot_animation(source_mitigation, sink_rate):
            C = self.run_simulation(1, source_mitigation, sink_rate)
            self.plot_concentration(C, source_mitigation, sink_rate)

    def simulation_movie(self):
        fig, ax = plt.subplots(figsize=(4, 3))

        ax.set_title("Contamination Spreading in a Watershed")
        C = self.run_simulation(1)
        im = ax.imshow(self.IC, cmap='Greys')#, vmin=0, vmax=self.IC.max(), cmap='Greys', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        time_text = ax.annotate('0 days', (0.02, 0.97), xycoords='axes fraction', va='top')
        fig.tight_layout()

        def update(t):
            C = self.run_simulation(t)
            im.set_array(C)
            im.set_clim(vmin=0, vmax=np.max(C))
            time_text.set_text(f'{365*t:.0f} days')
            return im,

        ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 100), interval=80, blit=True)
        plt.close()  # Prevent double display in Jupyter Notebook
        return HTML(ani.to_jshtml())