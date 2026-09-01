"""
Example custom BEC simulation extending the base class.

This example demonstrates how to create a custom simulation by inheriting
from BaseBEC and overriding specific methods.
"""
from src.models.base_BEC import BaseBEC
import torch
import numpy as np


class CustomBEC(BaseBEC):
    """
    Example custom BEC simulation.
    
    This example shows how to:
    - Add custom parameters
    - Modify the main simulation loop
    - Add custom measurements
    - Write custom outputs
    """
    
    def __init__(self, parameters, system, app, simulation_name):
        """
        Initialize the custom BEC simulation.
        """
        # Call parent constructor
        super().__init__(parameters, system, app, simulation_name)
        
        # Add any additional initialization here
        # These will be available before _initialize_custom_parameters() is called
    
    def _initialize_custom_parameters(self):
        """
        Initialize parameters specific to this custom simulation.
        """
        # Example: Extract custom parameters from the configuration
        self.my_special_time = self.parameters.get("special_event_time", 100)
        self.enable_custom_feature = self.parameters.get("enable_feature", False)
        
        # Example: Initialize custom measurement arrays
        self.custom_measurements = []
        self.phase_snapshots = {}
        
        self.logger.info(f"Custom parameters initialized. Special time: {self.my_special_time}")
    
    def _main_simulation_loop(self):
        """
        Custom main simulation loop with additional physics.
        
        This example shows:
        - Time-dependent modifications
        - Custom measurements
        - Conditional operations
        """
        count = 0
        event_triggered = False
        
        self.logger.info("Starting custom simulation loop...")
        
        # You could open custom diagnostic files here
        # diagnostic_file = open("custom_diagnostics.txt", "w")
        
        for iteration in range(self.kmax):
            # Current time
            t = self.dt * iteration * self.omega_ho
            
            # Total potential: interaction + external
            utot = self.u * torch.abs(self.psi) ** 2 + self.uext
            
            # Save data at regular intervals
            if iteration % (self.kmax // self.shots) == 0:
                self._write_iteration_data(count, t)
                
                # CUSTOM: Additional measurements at snapshot times
                self._measure_custom_observable(count)
                
                count += 1
            
            # CUSTOM: Apply special operation at specific time
            if iteration == self.my_special_time and not event_triggered:
                self.logger.info(f"Triggering special event at iteration {iteration}")
                self._apply_special_operation()
                event_triggered = True
            
            # CUSTOM: Example of time-dependent modification
            if self.enable_custom_feature and iteration > 500:
                # Example: gradually reduce interaction strength
                effective_u = self.u * (1.0 - 0.001 * (iteration - 500) / self.kmax)
                utot = effective_u * torch.abs(self.psi) ** 2 + self.uext
            
            # Perform time step
            self._step(utot, self.dtau, self.p_sq, self.d_x)
            
            # CUSTOM: Check for interesting phenomena every 100 iterations
            if iteration % 100 == 0:
                self._check_system_state(iteration)
        
        # Close any opened files
        # diagnostic_file.close()
        
        self.logger.info(f"Custom simulation loop completed. Total iterations: {self.kmax}")
    
    def _apply_special_operation(self):
        """
        Example of a custom operation to apply during simulation.
        
        This could be:
        - Phase imprinting
        - Sudden potential change
        - Wavefunction modification
        - etc.
        """
        # TODO: Implement your special operation
        # Example: imprint a phase
        # phase = self._calculate_some_phase()
        # self.psi = self.psi * torch.exp(1j * phase)
        
        self.logger.info("Special operation applied")
    
    def _measure_custom_observable(self, count):
        """
        Measure a custom observable at this time step.
        
        Args:
            count: int, snapshot counter
        """
        # TODO: Implement your custom measurement
        # Example: measure peak density
        density = self.get_density()
        peak_density = torch.max(density).item()
        
        self.custom_measurements.append({
            'count': count,
            'peak_density': peak_density,
            # Add more measurements as needed
        })
        
        # Example: save phase at specific times
        if count % 10 == 0:
            self.phase_snapshots[count] = self._extract_phase().cpu().numpy()
    
    def _check_system_state(self, iteration):
        """
        Check the state of the system and log or take action if needed.
        
        Args:
            iteration: int, current iteration number
        """
        # TODO: Implement system checks
        # Example: check normalization
        norm = self.get_normalization()
        if abs(norm - 1.0) > 0.01:
            self.logger.warning(f"Iteration {iteration}: Normalization drift detected: {norm:.6f}")
        
        # Example: check for numerical instabilities
        if torch.isnan(self.psi).any():
            self.logger.error(f"NaN detected at iteration {iteration}!")
            raise RuntimeError("Simulation became unstable")
    
    def _write_custom_outputs(self):
        """
        Write custom output files specific to this simulation.
        """
        # TODO: Write your custom outputs
        
        # Example: Save custom measurements
        if self.custom_measurements:
            import json
            with open('custom_measurements.json', 'w') as f:
                json.dump(self.custom_measurements, f, indent=2)
            self.logger.info("Custom measurements saved to custom_measurements.json")
        
        # Example: Save phase snapshots
        if self.phase_snapshots:
            np.savez('phase_snapshots.npz', **self.phase_snapshots)
            self.logger.info(f"Saved {len(self.phase_snapshots)} phase snapshots")
    
    # You can also add completely new methods specific to your simulation
    
    def _calculate_some_phase(self):
        """
        Example of a completely new method for this simulation.
        """
        # TODO: Implement your calculation
        return torch.zeros_like(self.psi, dtype=torch.float64)


# Example usage notes:
"""
To use this custom BEC class:

1. In your configuration file (configuration_file.json), add any custom parameters:
   {
       ...
       "special_event_time": 150,
       "enable_feature": true,
       ...
   }

2. In your simulation setup, use CustomBEC instead of BEC:
   
   from src.models.example_custom_BEC import CustomBEC
   
   # Inside your simulation loop:
   bec = CustomBEC(parameters, system, app, simulation_name)
   bec.evolve()

3. Extend or modify as needed for your specific physics!
"""
